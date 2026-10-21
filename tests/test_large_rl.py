import json
import pytest
import torch
from src.rl_train import group_policy_loss
from src.rl_positions import sample
from src.packed_data import encoded
from src.pipeline import file_hash
from src.vocab import ChessTokenizer

def test_within_position_reward_promotes_better_move():
    scores=torch.tensor([0.,0.],requires_grad=True)
    logs=scores.log_softmax(0)
    loss,kl=group_policy_loss(logs,logs.detach(),[0,1],[0.,-1.],.2)
    loss.backward()
    assert kl.item()==0 and scores.grad[0]<0<scores.grad[1]
    scores=torch.tensor([0.,0.],requires_grad=True);logs=scores.log_softmax(0)
    loss,_=group_policy_loss(logs,logs.detach(),[0,1],[-.5,-.5],.2)
    loss.backward();torch.testing.assert_close(scores.grad,torch.zeros(2))

def test_training_only_packed_position_sampling(tmp_path):
    tok=ChessTokenizer(); ids=[]
    for moves in [('e2e4','e7e5','g1f3','b8c6'),('d2d4','d7d5','c2c4','e7e6')]:
        ids.extend([0,*tok.encode(moves),1])
    (tmp_path/'train.bin').write_bytes(encoded(ids))
    (tmp_path/'manifest.json').write_text(json.dumps(dict(sha256={'train.bin':file_hash(tmp_path/'train.bin')},selection={'validation_sha256':'heldout'})))
    output=tmp_path/'positions.json';sample(tmp_path,output,count=5,seed=42)
    report=json.loads(output.read_text())
    assert report['count']==5 and report['excluded_validation_sha256']=='heldout'
    assert len({tuple(p['moves']) for p in report['positions']})==5
    import chess
    for position in report['positions']:
        board=chess.Board()
        for move in position['moves']:board.push_uci(move)
        assert board.fen()==position['fen']
    (tmp_path/'train.bin').write_bytes(encoded(ids+[0]))
    with pytest.raises(ValueError,match='hash differs'):
        sample(tmp_path,output,count=5)
