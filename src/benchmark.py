"""Measure one-move latency, throughput and CUDA memory on fixed positions."""
from __future__ import annotations
import argparse
import gc
import importlib.metadata
import statistics
import time
import torch
from src.pipeline import choose_device, evaluation_positions, file_hash, seed_all, write_json
from src.serving import TensorRTPolicy, TorchPolicy

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--positions", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--backend", choices=["pytorch", "tensorrt-llm", "both"], default="pytorch")
    p.add_argument("--hf-model")
    p.add_argument("--count", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--require-cuda", action="store_true")
    a = p.parse_args()
    if a.count < 1:
        p.error("count must be positive.")
    if a.backend != "pytorch" and not a.hf_model:
        p.error("--hf-model is required for TensorRT-LLM.")
    seed_all(a.seed)
    device = choose_device(a.require_cuda)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    positions = evaluation_positions(a.positions, a.count, a.seed)
    results = {}
    predictions = {}
    for backend in (["pytorch", "tensorrt-llm"] if a.backend == "both" else [a.backend]):
        policy = TorchPolicy(a.checkpoint, device) if backend == "pytorch" else TensorRTPolicy(a.hf_model)
        for position in positions[:min(5, len(positions))]:
            policy.move(position["moves"])
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        latencies, moves = [], []
        for position in positions:
            start = time.perf_counter()
            move = policy.move(position["moves"])
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - start)
            moves.append(move)
        predictions[backend] = moves
        ordered = sorted(latencies)
        results[backend] = {"count": len(moves), "median_latency_ms": statistics.median(latencies) * 1000,
                            "p95_latency_ms": ordered[min(len(ordered) - 1, int(.95 * len(ordered)))] * 1000,
                            "moves_per_second_serial": len(moves) / sum(latencies),
                            "torch_allocator_peak_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
                            "device_memory_used_bytes": (lambda pair: pair[1] - pair[0])(torch.cuda.mem_get_info()) if device.type == "cuda" else None,
                            "version": torch.__version__ if backend == "pytorch" else importlib.metadata.version("tensorrt_llm")}
        if hasattr(policy, "close"):
            policy.close()
        del policy
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    agreement = sum(left == right for left, right in zip(predictions["pytorch"], predictions["tensorrt-llm"])) / len(positions) if a.backend == "both" else None
    write_json(a.output, {"seed": a.seed, "positions_sha256": file_hash(a.positions),
                         "checkpoint_sha256": file_hash(a.checkpoint), "device": str(device),
                         "results": results, "move_agreement_fraction": agreement,
                         "memory_note": "Device memory is global used memory after warmup; allocator peak covers Torch-owned memory only.",
                         "predictions": predictions})
    print(results, flush=True)

if __name__ == "__main__":
    main()
