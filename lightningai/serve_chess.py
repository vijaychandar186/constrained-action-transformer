"""Host the playable chess app in its own CPU Studio."""
import json
import os
import shlex
from pathlib import Path,PurePosixPath
from lightningai.config import ROOT,initialize,teamspace
from lightningai.launch import package
from src.pipeline import write_json

def main():
    settings=initialize()
    from lightning_sdk import Studio,Machine
    studio=Studio('queens-gambit-play',teamspace=teamspace(),create_ok=True)
    archive=ROOT/'lightningai/package.zip';package(archive)
    studio.start(machine=Machine.CPU)
    studio.upload_file(str(archive),remote_path='queens-gambit-package.zip')
    remote=shlex.quote(settings['remote_dir'])
    studio.run(f'python -m zipfile -e ~/queens-gambit-package.zip ~/{remote}')
    studio.run('python -m pip install -q chess numpy')
    source=PurePosixPath('/teamspace/studios')/settings['studio']/settings['remote_dir']
    original=str(source/'runs/large/checkpoints/pretrain.pth')
    rl=str(source/'runs/large_rl/checkpoints/rl.pth')
    probe='from pathlib import Path,PurePosixPath; import torch; p=Path('+repr(original)+'); print("Original checkpoint available:",p.is_file()); assert p.is_file(), "Shared checkpoint path is unavailable"'
    studio.run('python -c '+shlex.quote(probe))
    command=(f'cd ~/{remote} && mkdir -p runs/serving && '
             f'python -u -m src.chess_web --host 0.0.0.0 --port 8000 '
             f'--original {shlex.quote(original)} --rl {shlex.quote(rl)} '
             '> runs/serving/chess.log 2>&1')
    output,code=studio.run_and_detach(command,timeout=15)
    if code is not None:
        raise RuntimeError('Chess server exited during startup.')
    health=studio.run('python -c '+shlex.quote('import urllib.request; print(urllib.request.urlopen("http://127.0.0.1:8000/health").read().decode())'))
    endpoints=studio.list_ports()
    found=[e for e in endpoints if '8000' in e.ports]
    endpoint=found[0] if found else studio.add_ports({'Chess':8000})[0]
    record=dict(state='serving',studio=studio.name,machine='CPU',port=8000,urls=endpoint.urls,
                auth_enabled=getattr(endpoint.auth,'enabled',None) if endpoint.auth else None,
                models=['original','rl'],checkpoint_source_studio=settings['studio'])
    write_json(ROOT/'lightningai/serving.json',record)
    print(json.dumps(record),flush=True)
    print('Health:',health,flush=True)

if __name__=='__main__':
    try:main()
    except Exception as error:
        key=os.environ.get('LIGHTNING_API_KEY','<unused>')
        print('Chess hosting failed:',str(getattr(error,'body',None) or error).replace(key,'[redacted]')[:600],flush=True)
        raise SystemExit(1) from None
