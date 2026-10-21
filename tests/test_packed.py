import json
from pathlib import Path
import numpy as np
import pytest
import torch
from src.model import GPTModel
from src.packed_data import PackedDataset, pack, encoded
from src.packed_train import parameter_count, train
from src.pipeline import file_hash

def pgn(moves,white=2000,black=2000):
    return f'[WhiteElo "{white}"]\n[BlackElo "{black}"]\n[Result "*"]\n\n{moves} *\n\n'

def test_packed_windows_count_each_target_once(tmp_path):
    path=tmp_path/'tokens.bin'; path.write_bytes(encoded(list(range(18))))
    data=PackedDataset(path,4,target_tokens=12)
    assert len(data)==3
    targets=torch.cat([data[i][1] for i in range(len(data))]).tolist()
    assert targets==list(range(1,13))
    with pytest.raises(ValueError):
        PackedDataset(path,4,target_tokens=20)

def test_stream_pack_filters_deduplicates_excludes_validation_and_resumes(tmp_path):
    validation=tmp_path/'validation.pgn'; validation.write_text(pgn('1. d4 d5 2. c4'))
    source=tmp_path/'train.pgn'; a=pgn('1. e4 e5 2. Nf3 Nc6'); b=pgn('1. c4 e5 2. Nc3')
    source.write_text(a+a+pgn('1. d4 d5 2. c4')+pgn('1. f4 e5',white=1000)+b)
    output=tmp_path/'packed'
    with pytest.raises(ValueError,match='Sources exhausted'):
        pack([str(source)],output,str(validation),12)
    # Append a third distinct game and resume the retained token file.
    source.write_text(source.read_text()+pgn('1. Nf3 d5 2. g3'))
    pack([str(source)],output,str(validation),12)
    m=json.loads((output/'manifest.json').read_text())
    assert m['training_games']==3 and m['training_tokens']==16
    assert m['validation_games']==1
    assert (output/'train.bin').stat().st_size==32
    before=(output/'train.bin').read_bytes()
    pack([str(source)],output,str(validation),12)
    assert (output/'train.bin').read_bytes()==before

def test_sdpa_preserves_causal_logits():
    config=dict(vocab_size=32,context_length=8,emb_dim=16,n_layers=2,n_heads=4,drop_rate=0.,qkv_bias=False)
    explicit=GPTModel(config).eval(); fused=GPTModel(dict(config,sdpa=True)).eval()
    fused.load_state_dict(explicit.state_dict())
    x=torch.randint(32,(2,8))
    torch.testing.assert_close(explicit(x),fused(x),rtol=1e-4,atol=1e-5)
    assert parameter_count()==24870144

def test_packed_training_time_limit_checkpoint_and_resume(tmp_path):
    data=tmp_path/'data'; data.mkdir()
    for name in ['train.bin','validation.bin']:
        (data/name).write_bytes(encoded([0,2,3,4,5,6,7,1]*5))
    manifest={'sha256':{n:file_hash(data/n) for n in ['train.bin','validation.bin']}}
    (data/'manifest.json').write_text(json.dumps(manifest))
    config=dict(vocab_size=4546,context_length=8,emb_dim=8,n_layers=1,n_heads=2,drop_rate=0.,qkv_bias=False,sdpa=True)
    profile=tmp_path/'profile.json'; profile.write_text(json.dumps(dict(config=config,batch_size=2,fits_budget=True)))
    output=tmp_path/'model.pth'
    with pytest.raises(SystemExit) as stopped:
        train(data,output,profile,config,32,max_hours=1e-12,require_cuda=False)
    assert stopped.value.code==124 and not output.exists()
    assert output.with_name('model_last.pth').exists()
    train(data,output,profile,config,32,max_hours=1,require_cuda=False)
    checkpoint=torch.load(output,map_location='cpu',weights_only=True)
    assert checkpoint['tokens_processed']==32 and checkpoint['step']==2
    assert json.loads((tmp_path/'model/metrics.json').read_text())['completed']
