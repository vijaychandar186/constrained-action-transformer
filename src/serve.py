"""Serve legal UCI moves from game history on a local HTTP endpoint."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from src.pipeline import choose_device
from src.serving import TorchPolicy, TensorRTPolicy

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint")
    p.add_argument("--hf-model")
    p.add_argument("--backend", choices=["pytorch", "tensorrt-llm"], default="pytorch")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    if a.backend == "pytorch" and not a.checkpoint or a.backend == "tensorrt-llm" and not a.hf_model:
        p.error("Supply --checkpoint for PyTorch or --hf-model for TensorRT-LLM.")
    policy = TorchPolicy(a.checkpoint, choose_device()) if a.backend == "pytorch" else TensorRTPolicy(a.hf_model)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/move":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 100000:
                    raise ValueError("Invalid request size.")
                request = json.loads(self.rfile.read(length))
                result, status = {"move": policy.move(request["moves"])}, 200
            except (ValueError, KeyError, TypeError) as error:
                result, status = {"error": str(error)}, 400
            body = json.dumps(result).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    server = HTTPServer((a.host, a.port), Handler)
    print(f"Serving /move at http://{a.host}:{a.port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        if hasattr(policy, "close"):
            policy.close()

if __name__ == "__main__":
    main()
