import json
import subprocess
import sys
import pytest
from src.comparison_report import summarize
from src.model import GPTModel
from src.pipeline import save_checkpoint

def test_match_labels_and_swapped_colors(tmp_path):
    config=dict(vocab_size=4546,context_length=16,emb_dim=8,n_heads=2,n_layers=1,drop_rate=0.,qkv_bias=False)
    checkpoint=tmp_path/'model.pth';save_checkpoint(checkpoint,GPTModel(config),config)
    games=tmp_path/'positions.pgn';games.write_text('[Result "*"]\n\n1. e4 e5 2. Nf3 Nc6 *\n')
    output=tmp_path/'matches.json'
    result=subprocess.run([sys.executable,'-m','src.compare','--candidate',str(checkpoint),'--candidate-name','Large','--baseline',str(checkpoint),'--baseline-name','Small','--positions',str(games),'--output',str(output),'--pairs','2','--max-plies','1'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    report=json.loads(output.read_text())
    assert report['draws']==4 and report['candidate_score_fraction']==.5
    assert report['paired_bootstrap_95_ci']==[.5,.5]
    assert report['truncated_games']==4
    pgn=output.with_suffix('.pgn').read_text()
    assert '[White "Large"]' in pgn and '[White "Small"]' in pgn
    assert 'Stockfish baseline' not in pgn

def test_paired_report_rejects_different_positions(tmp_path):
    base=dict(positions_sha256='same',seed=42,depth=8,engine={'name':'test'},decoding='greedy',mate_score=10000,count=1,
              checkpoint_sha256='a',rows=[dict(fen='same',moves=['e2e4'],centipawn_loss=100)])
    a=tmp_path/'a.json';b=tmp_path/'b.json';m=tmp_path/'m.json';out=tmp_path/'out.json'
    a.write_text(json.dumps(base));other=dict(base,checkpoint_sha256='b',rows=[dict(fen='different',moves=['e2e4'],centipawn_loss=0)])
    b.write_text(json.dumps(other));m.write_text('{}')
    with pytest.raises(ValueError,match='Sampled positions differ'):
        summarize(a,b,m,out)
