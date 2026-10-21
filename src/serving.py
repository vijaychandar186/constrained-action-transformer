"""One-move inference backends with a legal-move constraint."""
from __future__ import annotations
import json
from contextlib import nullcontext
from pathlib import Path
import chess
import torch
from src.pipeline import load_checkpoint, pick_move
from src.vocab import ChessTokenizer

def replay(moves):
    if not isinstance(moves, list) or not all(isinstance(move, str) for move in moves):
        raise ValueError("moves must be a list of UCI strings.")
    board = chess.Board()
    for uci in moves:
        board.push_uci(uci)
    if board.is_game_over(claim_draw=True):
        raise ValueError("The game is already over.")
    return board

class TorchPolicy:
    def __init__(self, checkpoint, device):
        self.model, self.tokenizer, _ = load_checkpoint(checkpoint, device)
        self.device = device

    def move(self, history):
        board = replay(history)
        return pick_move(self.model, board, history, self.tokenizer, self.device).uci()

class LegalLogits:
    """This callback applies only to a single generated move per request."""
    def __init__(self, allowed):
        self.allowed = allowed

    def __call__(self, req_id, logits, token_ids, stream_ptr=None, client_id=None):
        stream = torch.cuda.ExternalStream(stream_ptr) if stream_ptr else torch.cuda.current_stream() if logits.is_cuda else None
        with torch.cuda.stream(stream) if stream is not None else nullcontext():
            keep = torch.zeros(logits.shape[-1], dtype=torch.bool, device=logits.device)
            keep[self.allowed] = True
            logits.masked_fill_(~keep, -torch.inf)

class TensorRTPolicy:
    def __init__(self, model_dir):
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT-LLM requires a CUDA-capable Linux environment.")
        # GPT-2 uses AutoDeploy, since the native backend does not list it.
        from tensorrt_llm import LLM
        metadata = json.loads((Path(model_dir) / "chess_metadata.json").read_text())
        from src.boardstate import BoardTokenizer
        self.tokenizer = BoardTokenizer() if metadata["board_state"] else ChessTokenizer()
        if self.tokenizer.move_to_id != metadata["vocabulary"]:
            raise ValueError("Exported vocabulary differs from the chess tokenizer.")
        config = json.loads((Path(model_dir) / "config.json").read_text())
        self.context_length = config["n_positions"]
        self.llm = LLM(model=model_dir, backend="_autodeploy", max_batch_size=1,
                       max_seq_len=self.context_length + 1)

    def move(self, history):
        from tensorrt_llm import SamplingParams
        board = replay(history)
        allowed = [self.tokenizer.move_to_id[move.uci()] for move in board.legal_moves]
        tokens = self.tokenizer.context_ids(history) if hasattr(self.tokenizer, "context_ids") else [self.tokenizer.game_start_id] + self.tokenizer.encode(history)
        tokens = tokens[-self.context_length:]
        params = SamplingParams(max_tokens=1, top_k=1, temperature=1,
                                logits_processor=LegalLogits(allowed))
        outputs = self.llm.generate([tokens], params)
        token = outputs[0].outputs[0].token_ids[0]
        move = self.tokenizer.id_to_move[token]
        if chess.Move.from_uci(move) not in board.legal_moves:
            raise RuntimeError("Serving backend violated the legal-move constraint.")
        return move

    def close(self):
        self.llm.shutdown()
