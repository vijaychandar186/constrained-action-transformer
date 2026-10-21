import json
import threading
import urllib.request
import urllib.error
import pytest
import torch
from src.chess_web import make_server
from src.serving import TorchPolicy
from src.model import GPTModel
from src.pipeline import save_checkpoint

@pytest.fixture
def web(tmp_path):
    checkpoint=tmp_path/'model.pth'
    cfg=dict(vocab_size=4546,context_length=16,emb_dim=8,n_heads=2,n_layers=1,drop_rate=.8,qkv_bias=False)
    save_checkpoint(checkpoint,GPTModel(cfg),cfg)
    server=make_server('127.0.0.1',0,{'original':('Original model',checkpoint),'rl':('RL model',tmp_path/'pending.pth')})
    worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    yield 'http://127.0.0.1:'+str(server.server_port),checkpoint
    server.shutdown();server.server_close();worker.join()

def test_web_board_catalog_and_legal_move(web):
    url,_=web
    assert b'Your move.' in urllib.request.urlopen(url).read()
    assert b'export' in urllib.request.urlopen(url+'/vendor/chess.js').read()
    catalog=json.load(urllib.request.urlopen(url+'/api/models'))
    assert [m['id'] for m in catalog['models']]==['original']
    request=urllib.request.Request(url+'/api/move',data=json.dumps({'moves':['e2e4'],'model':'original'}).encode(),headers={'Content-Type':'application/json'})
    result=json.load(urllib.request.urlopen(request))
    import chess
    board=chess.Board();board.push_uci('e2e4')
    assert chess.Move.from_uci(result['move']) in board.legal_moves
    bad=urllib.request.Request(url+'/api/move',data=b'{"moves":["e2e5"]}',headers={'Content-Type':'application/json'})
    with pytest.raises(urllib.error.HTTPError) as error:urllib.request.urlopen(bad)
    assert error.value.code==400
    with pytest.raises(urllib.error.HTTPError):urllib.request.urlopen(url+'/.env')

def test_inference_turns_dropout_off(web):
    _,checkpoint=web
    policy=TorchPolicy(checkpoint,'cpu')
    assert not policy.model.training
    ids=torch.tensor([[0,2,3]])
    with torch.no_grad():torch.testing.assert_close(policy.model(ids),policy.model(ids))
