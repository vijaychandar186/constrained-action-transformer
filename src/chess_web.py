"""Serve a browser chess board and legal model moves from one origin."""
import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler,HTTPServer
from pathlib import Path
from urllib.parse import urlsplit
import torch
from src.pipeline import choose_device
from src.serving import TorchPolicy

ROOT=Path(__file__).resolve().parents[1]

def make_server(host,port,models,policies=None,static=ROOT/'ui'):
    policies={} if policies is None else policies
    def available():
        return [{'id':key,'name':name} for key,(name,path) in models.items() if Path(path).is_file()]
    class Handler(BaseHTTPRequestHandler):
        def reply(self,status,value):
            body=json.dumps(value).encode()
            self.send_response(status);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store')
            self.end_headers();self.wfile.write(body)
        def do_GET(self):
            path=urlsplit(self.path).path
            if path=='/api/models':
                return self.reply(200,{'models':available()})
            if path=='/health':
                return self.reply(200,{'status':'ok','models':available()})
            files={'/':'index.html','/app.js':'app.js','/style.css':'style.css','/vendor/chess.js':'vendor/chess.js','/vendor/chess.js.LICENSE':'vendor/chess.js.LICENSE'}
            if path not in files:
                return self.reply(404,{'error':'Not found'})
            file=static/files[path]
            if not file.is_file():
                return self.reply(404,{'error':'Not found'})
            body=file.read_bytes();self.send_response(200)
            kind='text/javascript' if file.suffix=='.js' else mimetypes.guess_type(str(file))[0] or 'application/octet-stream'
            self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)))
            self.end_headers();self.wfile.write(body)
        def do_POST(self):
            if urlsplit(self.path).path!='/api/move':
                return self.reply(404,{'error':'Not found'})
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=64000:
                    raise ValueError('Invalid request size.')
                request=json.loads(self.rfile.read(size))
                if not isinstance(request,dict):
                    raise ValueError('Expected a JSON object.')
                key=request.get('model','original')
                if key not in models or not Path(models[key][1]).is_file():
                    return self.reply(409,{'error':'This model is not ready yet. Choose another model.'})
                moves=request['moves']
                if not isinstance(moves,list) or len(moves)>1000 or any(not isinstance(v,str) or len(v) not in (4,5) for v in moves):
                    raise ValueError('Invalid game history.')
                if key not in policies:
                    policies[key]=TorchPolicy(str(models[key][1]),choose_device())
                move=policies[key].move(moves)
                return self.reply(200,{'move':move,'model':key})
            except (ValueError,KeyError,TypeError) as error:
                return self.reply(400,{'error':str(error)})
            except Exception:
                return self.reply(503,{'error':'The model could not respond. Please retry.'})
    return HTTPServer((host,port),Handler)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original',required=True);p.add_argument('--rl')
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8000)
    a=p.parse_args();torch.set_num_threads(min(4,torch.get_num_threads()))
    models={'original':('Original large model',a.original)}
    if a.rl:models['rl']=('Stockfish RL model',a.rl)
    server=make_server(a.host,a.port,models)
    print(f'Chess app listening on {a.host}:{a.port}',flush=True)
    try:server.serve_forever()
    finally:server.server_close()

if __name__=='__main__':main()
