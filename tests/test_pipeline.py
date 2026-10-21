import json
import os
import subprocess
import sys
from pathlib import Path
import chess
import pytest
import torch
from src.model import GPTModel
from src.pipeline import legal_distribution, load_checkpoint, save_checkpoint
from src.rl_train import policy_loss
from src.vocab import ChessTokenizer
from src.boardstate import BoardTokenizer, create_board_loader
from src.serving import LegalLogits, replay

@pytest.fixture
def small_model():
    torch.manual_seed(7)
    tokenizer = ChessTokenizer()
    config = dict(vocab_size=tokenizer.vocab_size, context_length=16, emb_dim=16,
                  n_heads=4, n_layers=1, drop_rate=0.0, qkv_bias=False)
    return GPTModel(config).eval(), tokenizer, config

def test_causal_attention_does_not_see_future(small_model):
    model, tokenizer, _ = small_model
    left = torch.tensor([[2, 3, 4, 5]])
    right = torch.tensor([[2, 3, 8, 9]])
    with torch.no_grad():
        torch.testing.assert_close(model(left)[:, :2], model(right)[:, :2])

@pytest.mark.parametrize("fen", [chess.STARTING_FEN, "7k/4P3/8/8/8/8/8/K7 w - - 0 1"])
def test_distribution_contains_only_legal_moves(small_model, fen):
    model, tokenizer, _ = small_model
    board = chess.Board(fen)
    moves, logs = legal_distribution(model, board, [], tokenizer, "cpu")
    assert set(moves) == set(board.legal_moves)
    torch.testing.assert_close(logs.exp().sum(), torch.tensor(1.0))
    assert torch.isfinite(logs).all()

def test_reference_kl_and_policy_gradient_direction():
    scores = torch.tensor([0.0, 0.0], requires_grad=True)
    logs = scores.log_softmax(0)
    reference = logs.detach().clone()
    loss, kl = policy_loss(logs, reference, 0, 1.0, .1)
    assert kl.item() == 0
    loss.backward()
    assert scores.grad[0] < 0 < scores.grad[1]

def test_checkpoint_keeps_architecture_and_vocabulary(small_model, tmp_path):
    model, tokenizer, config = small_model
    path = tmp_path / "model.pth"
    save_checkpoint(path, model, config)
    restored, decoded, metadata = load_checkpoint(path)
    assert metadata["config"] == config
    assert decoded.move_to_id == tokenizer.move_to_id
    inputs = torch.tensor([[0, tokenizer.move_to_id["e2e4"]]])
    with torch.no_grad():
        torch.testing.assert_close(model(inputs), restored(inputs))

def test_piece_targets_are_excluded_from_loss():
    import chess.pgn
    game = chess.pgn.Game()
    node = game
    for uci in ["e2e4", "e7e5", "g1f3"]:
        node = node.add_variation(chess.Move.from_uci(uci))
    tokenizer = BoardTokenizer()
    loader = create_board_loader([game], tokenizer, 65, 2, False)
    targets = torch.cat([batch[1].flatten() for batch in loader])
    assert (targets == -100).any()
    assert (targets != -100).any()
    assert torch.all(targets[targets != -100] < tokenizer.move_vocab_size)

def test_logits_callback_masks_special_and_illegal_tokens():
    logits = torch.arange(10, dtype=torch.float).reshape(1, 1, 10)
    LegalLogits([2, 7])(0, logits, [[]])
    assert logits[0, 0, 2] == 2 and logits[0, 0, 7] == 7
    assert torch.isneginf(logits[0, 0, [0, 1, 3, 4, 5, 6, 8, 9]]).all()

def test_serving_rejects_illegal_history():
    with pytest.raises(ValueError):
        replay(["e2e5"])
    assert replay(["e2e4", "e7e5"]).fullmove_number == 2

def test_hf_export_preserves_logits(small_model):
    pytest.importorskip("transformers")
    from src.export_hf import convert
    model, tokenizer, config = small_model
    exported = convert(model, config, tokenizer)
    inputs = torch.tensor([[0, 2, 47, 103]])
    with torch.no_grad():
        torch.testing.assert_close(model(inputs), exported(inputs, use_cache=False).logits, rtol=2e-4, atol=2e-5)

def test_queue_stops_and_resumes_without_repeating_completed_stage(tmp_path):
    root = Path(__file__).resolve().parents[1]
    marker, allow, final = (tmp_path / name for name in ("first", "allow", "final"))
    plan = {"stages": [
        {"name": "first", "command": ["python", "-c", f"from pathlib import Path; p=Path({str(marker)!r}); p.write_text(p.read_text()+'x' if p.exists() else 'x')"]},
        {"name": "second", "command": ["python", "-c", f"from pathlib import Path; import sys; sys.exit(0 if Path({str(allow)!r}).exists() else 2)"]},
        {"name": "third", "command": ["python", "-c", f"from pathlib import Path; Path({str(final)!r}).touch()"]}
    ]}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    run_dir = tmp_path / "run"
    command = [sys.executable, str(root / "scripts/run_experiment_queue.py"), "--plan", str(plan_path), "--run-dir", str(run_dir)]
    first = subprocess.run(command, capture_output=True, text=True)
    assert first.returncode != 0
    assert marker.read_text() == "x" and not final.exists()
    status = json.loads((run_dir / "status.json").read_text())
    assert status["stages"]["second"]["state"] == "failed"
    allow.touch()
    second = subprocess.run(command + ["--resume"], capture_output=True, text=True)
    assert second.returncode == 0, second.stderr
    assert marker.read_text() == "x" and final.exists()
    assert json.loads((run_dir / "status.json").read_text())["state"] == "completed"


def test_lock_check_does_not_signal_current_process():
    from scripts.run_experiment_queue import process_alive
    assert process_alive(os.getpid())

def test_queue_times_out_and_does_not_run_next_stage(tmp_path):
    root = Path(__file__).resolve().parents[1]
    marker = tmp_path / "should_not_exist"
    plan = {"stages": [
        {"name": "slow", "command": ["python", "-c", "import time; time.sleep(10)"], "timeout_seconds": .1},
        {"name": "next", "command": ["python", "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"]}
    ]}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    run_dir = tmp_path / "run"
    result = subprocess.run([sys.executable, str(root / "scripts/run_experiment_queue.py"), "--plan", str(plan_path), "--run-dir", str(run_dir)], capture_output=True, text=True, timeout=15)
    assert result.returncode != 0 and not marker.exists()
    state = json.loads((run_dir / "status.json").read_text())
    assert state["stages"]["slow"]["error"] == "Stage exceeded timeout."
    assert not (run_dir / ".queue.lock").exists()


def test_lightning_debug_disabled_after_sdk_import(monkeypatch):
    pytest.importorskip("lightning_sdk")
    from lightning_sdk.lightning_cloud import env
    from lightningai.config import initialize
    import http.client
    monkeypatch.setattr(env, "DEBUG", True)
    monkeypatch.setenv("LIGHTNING_API_KEY", "test-placeholder")
    initialize()
    assert env.DEBUG is False
    assert http.client.HTTPConnection.debuglevel == 0
    assert http.client.HTTPSConnection.debuglevel == 0


def test_upload_package_has_linux_shell_scripts_and_no_credentials(tmp_path):
    import zipfile
    from lightningai.launch import package
    destination = tmp_path / "package.zip"
    package(destination)
    with zipfile.ZipFile(destination) as archive:
        assert b"\r" not in archive.read("lightningai/bootstrap.sh")
        assert not any(Path(name).name == ".env" for name in archive.namelist())
