"""
Chess-specific inference: legal move masking.

This is the key difference from standard LLM generation.

The model outputs logits over the FULL vocab (all ~4546 moves).
We zero out every illegal move for the current board position,
then renormalize — the model's learned distribution is preserved
within the legal subset.

generate_move()        — pick one move (used during a game)
generate_sample_game() — generate N moves from a fresh board (for eval during training)
play_game()            — full interactive loop using python-chess
"""

import torch
import chess

from src.vocab import ChessTokenizer, GAME_START


def _legal_move_ids(board: chess.Board, tokenizer: ChessTokenizer) -> list[int]:
    """Return token ids for all legal moves on the board."""
    ids = []
    for move in board.legal_moves:
        uci = move.uci()
        if uci in tokenizer.move_to_id:
            ids.append(tokenizer.move_to_id[uci])
    return ids


def generate_move(
    model,
    board: chess.Board,
    context_ids: list[int],
    tokenizer: ChessTokenizer,
    device,
    temperature: float = 1.0,
) -> str:
    """
    Predict the next move for the current board position.

    1. Run the model on the current game context.
    2. Get logits for the next token.
    3. Mask all non-legal moves to -inf.
    4. Sample from the remaining distribution.

    Returns the chosen move as a UCI string.
    """
    model.eval()

    ctx_len = model.pos_emb.weight.shape[0]
    # Trim to context window
    if hasattr(tokenizer, "context_ids"):
        context_ids = tokenizer.context_ids([move.uci() for move in board.move_stack])
    ids = context_ids[-ctx_len:]
    idx = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, seq)

    with torch.no_grad():
        logits = model(idx)                     # (1, seq, vocab_size)

    logits = logits[0, -1, :]                   # (vocab_size,)

    # Mask illegal moves
    legal_ids = _legal_move_ids(board, tokenizer)
    if not legal_ids:
        raise ValueError("No legal moves — game should be over")

    mask = torch.full_like(logits, float("-inf"))
    mask[legal_ids] = logits[legal_ids]

    # Temperature scaling + sample
    probs    = torch.softmax(mask / temperature, dim=-1)
    next_id  = torch.multinomial(probs, num_samples=1).item()

    return tokenizer.id_to_move[next_id]


def generate_sample_game(
    model,
    tokenizer: ChessTokenizer,
    device,
    num_moves: int = 20,
    temperature: float = 1.0,
) -> list[str]:
    """
    Generate a sequence of moves from a fresh board.
    Used during training to sanity-check the model.
    Returns a list of UCI move strings.
    """
    board       = chess.Board()
    context_ids = [tokenizer.game_start_id]
    moves_made  = []

    for _ in range(num_moves):
        if board.is_game_over():
            break
        try:
            uci = generate_move(model, board, context_ids, tokenizer, device, temperature)
        except ValueError:
            break

        move = chess.Move.from_uci(uci)
        board.push(move)
        context_ids.append(tokenizer.move_to_id[uci])
        moves_made.append(uci)

    return moves_made


def play_game(
    model,
    tokenizer: ChessTokenizer,
    device,
    model_plays: str = "both",
    temperature: float = 1.0,
):
    """
    Interactive game loop.

    model_plays: "white" | "black" | "both"
      "white"  — model plays white, you play black (enter UCI moves)
      "black"  — you play white, model plays black
      "both"   — model plays both sides (watch a full auto game)
    """
    board       = chess.Board()
    context_ids = [tokenizer.game_start_id]
    move_num    = 1

    print(board)
    print()

    while not board.is_game_over():
        side = "white" if board.turn == chess.WHITE else "black"
        model_turn = (
            model_plays == "both"
            or (model_plays == "white" and board.turn == chess.WHITE)
            or (model_plays == "black" and board.turn == chess.BLACK)
        )

        if model_turn:
            uci = generate_move(model, board, context_ids, tokenizer, device, temperature)
            print(f"  Model ({side}): {uci}")
        else:
            legal_ucis = {m.uci() for m in board.legal_moves}
            while True:
                uci = input(f"  Your move ({side}, UCI): ").strip()
                if uci in legal_ucis:
                    break
                print(f"  Illegal. Legal moves: {sorted(legal_ucis)}")

        move = chess.Move.from_uci(uci)
        board.push(move)
        context_ids.append(tokenizer.move_to_id[uci])

        if board.turn == chess.WHITE:
            move_num += 1

        print(board)
        print()

    result = board.result()
    print(f"Game over: {result}")
    return result


if __name__ == "__main__":
    import sys
    from config import CHESS_GPT_CONFIG

    if len(sys.argv) < 2:
        print("Usage: python -m src.generate <checkpoint.pth> [white|black|both]")
        sys.exit(1)

    ckpt_path  = sys.argv[1]
    model_side = sys.argv[2] if len(sys.argv) > 2 else "both"
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = ChessTokenizer()
    from src.pipeline import load_checkpoint
    model, tokenizer, _ = load_checkpoint(ckpt_path, device)
    model.eval()

    play_game(model, tokenizer, device, model_plays=model_side)
