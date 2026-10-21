"""Board-conditioned move training with loss only on move targets."""
from __future__ import annotations
import torch
from torch.utils.data import DataLoader
import chess
from src.dataset import ChessDataset
from src.vocab import ChessTokenizer

PIECES = ".PNBRQKpnbrqk"

class BoardTokenizer(ChessTokenizer):
    def __init__(self):
        super().__init__()
        self.move_vocab_size = self.vocab_size
        for piece in PIECES:
            token = "<piece:" + piece + ">"
            index = len(self.move_to_id)
            self.move_to_id[token] = index
            self.id_to_move[index] = token
        self.vocab_size = len(self.move_to_id)

    def board_ids(self, board):
        return [self.move_to_id["<piece:" + (board.piece_at(square).symbol() if board.piece_at(square) else ".") + ">"]
                for square in chess.SQUARES]

    def context_ids(self, moves):
        board = chess.Board()
        ids = [self.game_start_id] + self.board_ids(board)
        for uci in moves:
            ids.append(self.move_to_id[uci])
            board.push_uci(uci)
            ids.extend(self.board_ids(board))
        return ids

def create_board_loader(games, tokenizer, context_length, batch_size, shuffle):
    if context_length < 65:
        raise ValueError("Board-state contexts need at least 65 tokens.")
    ids = []
    for game in games:
        if game.board().fen() != chess.Board().fen():
            raise ValueError("Board-state training currently requires standard initial positions.")
        ids.extend(tokenizer.context_ids([move.uci() for move in game.mainline_moves()]))
        ids.append(tokenizer.game_end_id)
    dataset = ChessDataset(ids, context_length, context_length // 2)
    for targets in dataset.target_ids:
        targets[(targets < 2) | (targets >= tokenizer.move_vocab_size)] = -100
    keep = [i for i, targets in enumerate(dataset.target_ids) if (targets != -100).any()]
    dataset.input_ids = [dataset.input_ids[i] for i in keep]
    dataset.target_ids = [dataset.target_ids[i] for i in keep]
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
