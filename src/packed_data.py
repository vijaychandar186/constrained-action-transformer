"""Prepare deduplicated move tokens on disk without retaining PGN trees."""
from __future__ import annotations
import argparse
import array
import hashlib
import io
import json
import os
import sqlite3
import sys
import urllib.request
from pathlib import Path
import chess
import chess.pgn
import numpy as np
import torch
import zstandard
from src.pipeline import file_hash, write_json
from src.vocab import ChessTokenizer

class MoveVisitor(chess.pgn.BaseVisitor):
    def __init__(self, tokenizer, min_elo=0, max_elo=10000):
        self.tokenizer, self.min_elo, self.max_elo = tokenizer, min_elo, max_elo
        self.headers, self.ids, self.errors = {}, [], []
    def visit_header(self, name, value):
        self.headers[name] = value
    def end_headers(self):
        try:
            ratings = [int(self.headers.get(k, 0)) for k in ('WhiteElo', 'BlackElo')]
            eligible = min(ratings) >= self.min_elo and max(ratings) <= self.max_elo
        except ValueError:
            eligible = False
        if not eligible or self.headers.get('Variant', 'Standard') not in ('Standard', 'Chess') or 'FEN' in self.headers:
            return chess.pgn.SKIP
    def begin_variation(self):
        return chess.pgn.SKIP
    def visit_move(self, board, move):
        self.ids.append(self.tokenizer.move_to_id[move.uci()])
    def handle_error(self, error):
        self.errors.append(str(error))
    def result(self):
        return self.ids if not self.errors else []

def encoded(ids):
    values = array.array('H', ids)
    if sys.byteorder != 'little':
        values.byteswap()
    return values.tobytes()

class PackedDataset(torch.utils.data.Dataset):
    def __init__(self, path, context_length=256, target_tokens=None):
        self.path, self.context_length = str(path), context_length
        count = Path(path).stat().st_size // 2
        self.count = (count - 1) // context_length
        if target_tokens is not None:
            required = (target_tokens + context_length - 1) // context_length
            if self.count < required:
                raise ValueError('Packed dataset does not contain the requested fresh-token budget.')
            self.count = required
        if self.count < 1:
            raise ValueError('Packed dataset is too short.')
        self.tokens = np.memmap(path, dtype='<u2', mode='r')
    def __len__(self):
        return self.count
    def __getitem__(self, index):
        start = index * self.context_length
        chunk = torch.from_numpy(np.array(self.tokens[start:start+self.context_length+1], dtype=np.int64))
        return chunk[:-1], chunk[1:]

def pack(sources, output, validation_pgn, target_tokens, min_elo=1800, max_elo=2500):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'manifest.json'
    policy = dict(sources=sources, target_tokens=target_tokens, min_elo=min_elo, max_elo=max_elo,
                  validation_sha256=file_hash(validation_pgn))
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved['selection'] != policy:
            raise ValueError('Existing packed dataset has different selection settings.')
        for name in ('train.bin', 'validation.bin'):
            if file_hash(output/name) != saved['sha256'][name]:
                raise ValueError('Packed data hash mismatch: '+name)
        print('Using verified packed dataset.', flush=True)
        return
    tokenizer = ChessTokenizer()
    excluded, validation = set(), bytearray()
    visitor = lambda: MoveVisitor(tokenizer)
    with open(validation_pgn, encoding='utf-8') as handle:
        while (ids := chess.pgn.read_game(handle, Visitor=visitor)) is not None:
            if ids:
                excluded.add(hashlib.sha256(encoded(ids)).digest())
                validation.extend(encoded([0, *ids, 1]))
    if not excluded:
        raise ValueError('A nonempty independent validation PGN is required.')
    (output/'validation.bin').write_bytes(validation)
    tokenizer.save(str(output/'vocab.txt'))
    database = sqlite3.connect(output/'games.sqlite')
    database.execute('CREATE TABLE IF NOT EXISTS games (hash BLOB PRIMARY KEY) WITHOUT ROWID')
    database.execute('CREATE TABLE IF NOT EXISTS progress (id INTEGER PRIMARY KEY, value TEXT)')
    row = database.execute('SELECT value FROM progress WHERE id=1').fetchone()
    progress = json.loads(row[0]) if row else dict(selection=policy, tokens=0, games=0, scanned=0)
    if progress['selection'] != policy:
        raise ValueError('Partial dataset selection changed.')
    path = output/'train.partial.bin'
    if not path.exists() and (output/'train.bin').exists():
        (output/'train.bin').replace(path)
    mode = 'r+b' if path.exists() else 'w+b'
    with open(path, mode) as destination:
        if destination.seek(0, 2) < progress['tokens']*2:
            raise ValueError('Partial data is shorter than its committed progress.')
        destination.truncate(progress['tokens']*2)
        destination.seek(0, 2)
        def commit():
            destination.flush()
            os.fsync(destination.fileno())
            database.execute('INSERT OR REPLACE INTO progress VALUES (1, ?)', (json.dumps(progress),))
            database.commit()
            write_json(output/'progress.json', progress)
        for source in sources:
            if progress['tokens'] >= target_tokens+1:
                break
            raw = urllib.request.urlopen(urllib.request.Request(source, headers={'User-Agent':'QueensGambit/1.0'}), timeout=120) if source.startswith('https://') else open(source,'rb')
            stream = zstandard.ZstdDecompressor().stream_reader(raw) if source.endswith('.zst') else raw
            with raw, io.TextIOWrapper(stream, encoding='utf-8', errors='replace') as handle:
                visit = lambda: MoveVisitor(tokenizer, min_elo, max_elo)
                while progress['tokens'] < target_tokens+1:
                    ids = chess.pgn.read_game(handle, Visitor=visit)
                    if ids is None:
                        break
                    progress['scanned'] += 1
                    if not ids:
                        continue
                    digest = hashlib.sha256(encoded(ids)).digest()
                    if digest in excluded:
                        continue
                    inserted = database.execute('INSERT OR IGNORE INTO games VALUES (?)', (digest,)).rowcount
                    if not inserted:
                        continue
                    destination.write(encoded([0,*ids,1]))
                    progress['tokens'] += len(ids)+2
                    progress['games'] += 1
                    if progress['games'] % 10000 == 0:
                        commit()
                        print(json.dumps(progress), flush=True)
            commit()
        commit()
    database.close()
    if progress['tokens'] < target_tokens+1:
        raise ValueError('Sources exhausted before the requested token budget; partial data retained.')
    path.replace(output/'train.bin')
    manifest = dict(selection=policy, training_tokens=progress['tokens'], training_games=progress['games'],
                    validation_tokens=len(validation)//2, validation_games=len(excluded), dtype='little-endian uint16',
                    sha256={n:file_hash(output/n) for n in ('train.bin','validation.bin')})
    write_json(manifest_path, manifest)
    print(json.dumps(manifest), flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', nargs='+', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--validation-pgn', required=True)
    p.add_argument('--target-tokens', type=int, required=True)
    p.add_argument('--min-elo', type=int, default=1800)
    p.add_argument('--max-elo', type=int, default=2500)
    a=p.parse_args()
    if a.target_tokens < 1 or not 0 <= a.min_elo <= a.max_elo:
        p.error('Use a positive token budget and ordered rating bounds.')
    pack(a.source,a.output,a.validation_pgn,a.target_tokens,a.min_elo,a.max_elo)

if __name__ == '__main__':
    main()
