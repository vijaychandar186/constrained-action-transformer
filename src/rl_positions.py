"""Sample RL positions from the verified training-only packed token stream."""
import argparse
import hashlib
import json
import random
from pathlib import Path
import chess
import numpy as np
from src.pipeline import file_hash,write_json
from src.vocab import ChessTokenizer

def sample(data,output,count=4096,seed=314):
    data=Path(data)
    manifest=json.loads((data/'manifest.json').read_text())
    source=data/'train.bin'
    if file_hash(source)!=manifest['sha256']['train.bin']:
        raise ValueError('Training token file hash differs from its manifest.')
    tokens=np.memmap(source,dtype='<u2',mode='r')
    tokenizer=ChessTokenizer(); rng=random.Random(seed)
    positions=[]; seen=set(); attempts=0
    while len(positions)<count and attempts<count*100:
        attempts+=1
        offset=rng.randrange(1,len(tokens))
        if int(tokens[offset])<2:
            continue
        left=max(0,offset-4096)
        starts=np.flatnonzero(tokens[left:offset]==tokenizer.game_start_id)
        if not len(starts):
            continue
        start=left+int(starts[-1])+1
        ids=tuple(int(t) for t in tokens[start:offset])
        if any(t<2 for t in ids) or ids in seen:
            continue
        board=chess.Board(); moves=[]
        for token in ids:
            uci=tokenizer.id_to_move[token]
            board.push_uci(uci); moves.append(uci)
        if board.is_game_over(claim_draw=True) or board.legal_moves.count()<2:
            continue
        seen.add(ids)
        positions.append(dict(fen=board.fen(),moves=moves))
    if len(positions)!=count:
        raise ValueError('Could not sample enough distinct nonterminal training positions.')
    write_json(output,dict(source_sha256=manifest['sha256']['train.bin'],
                           excluded_validation_sha256=manifest['selection']['validation_sha256'],
                           seed=seed,count=count,positions=positions))
    print('Training-only RL positions saved:',len(positions),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--count',type=int,default=4096);p.add_argument('--seed',type=int,default=314)
    a=p.parse_args();sample(a.data,a.output,a.count,a.seed)

if __name__=='__main__':
    main()
