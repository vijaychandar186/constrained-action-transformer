"""Single-pass training on fresh packed move tokens, with a measured T4 budget."""
from __future__ import annotations
import argparse
import json
import math
import signal
import time
from pathlib import Path
import torch
from src.model import GPTModel
from src.packed_data import PackedDataset
from src.pipeline import choose_device, file_hash, save_checkpoint, seed_all, write_json
from src.vocab import ChessTokenizer

CONFIG = dict(vocab_size=4546, context_length=256, emb_dim=384, n_heads=8,
              n_layers=12, drop_rate=.1, qkv_bias=False, sdpa=True)

def parameter_count(config=CONFIG):
    with torch.device('meta'):
        model=GPTModel(config)
    return sum(p.numel() for p in model.parameters())

def profile(output, config=CONFIG, max_hours=20):
    device=choose_device(True)
    seed_all(42)
    torch.set_num_threads(min(8,torch.get_num_threads()))
    results=[]
    for batch in (32,64):
        model=GPTModel(config).to(device).train()
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.1)
        scaler=torch.amp.GradScaler('cuda')
        inputs=torch.randint(config['vocab_size'],(batch,config['context_length']),device=device)
        targets=torch.randint(config['vocab_size'],inputs.shape,device=device)
        try:
            torch.cuda.reset_peak_memory_stats()
            times=[]
            for step in range(13):
                torch.cuda.synchronize()
                started=time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=device.type=='cuda'):
                    loss=torch.nn.functional.cross_entropy(model(inputs).flatten(0,1),targets.flatten())
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                scaler.step(optimizer)
                scaler.update()
                torch.cuda.synchronize()
                if step>=3:
                    times.append(time.monotonic()-started)
            speed=batch*config['context_length']/(sum(times)/len(times))
            results.append(dict(batch_size=batch,tokens_per_second=speed,peak_bytes=torch.cuda.max_memory_allocated()))
        except torch.cuda.OutOfMemoryError:
            results.append(dict(batch_size=batch,error='CUDA out of memory'))
        del model,optimizer,scaler,inputs,targets
        torch.cuda.empty_cache()
    viable=[r for r in results if 'tokens_per_second' in r]
    if not viable:
        raise RuntimeError('Neither benchmark batch size fits.')
    best=max(viable,key=lambda r:r['tokens_per_second'])
    tokens=20*parameter_count(config)
    estimate=tokens/best['tokens_per_second']*1.25
    report=dict(config=config,parameters=parameter_count(config),target_tokens=tokens,measurements=results,
                batch_size=best['batch_size'],estimated_training_hours=estimate/3600,
                max_training_hours=max_hours,fits_budget=estimate<=max_hours*3600,
                note='Synthetic steady-state batches; includes 25% allowance but excludes CPU data preparation.')
    write_json(output,report)
    print(json.dumps(report),flush=True)
    return report

def train(data, output, profile_path, config=CONFIG, target_tokens=None, max_hours=20, save_every=500, require_cuda=True):
    seed_all(42)
    device=choose_device(require_cuda)
    torch.set_num_threads(min(8,torch.get_num_threads()))
    budget=target_tokens or 20*parameter_count(config)
    profile_report=json.loads(Path(profile_path).read_text())
    if profile_report['config'] != config:
        raise ValueError('Benchmark and training architectures differ.')
    if not profile_report['fits_budget']:
        raise ValueError('Benchmark exceeds the authorized training budget; resize before training.')
    data=Path(data)
    manifest=json.loads((data/'manifest.json').read_text())
    for name in ('train.bin','validation.bin'):
        if file_hash(data/name)!=manifest['sha256'][name]:
            raise ValueError('Packed data hash mismatch.')
    dataset=PackedDataset(data/'train.bin',config['context_length'],budget)
    validation=PackedDataset(data/'validation.bin',config['context_length'])
    batch=profile_report['batch_size']
    loader=torch.utils.data.DataLoader(dataset,batch_size=batch,shuffle=True,num_workers=0,pin_memory=True,
                                      generator=torch.Generator().manual_seed(42))
    val_loader=torch.utils.data.DataLoader(validation,batch_size=batch,num_workers=0,pin_memory=True)
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    directory=output.parent/output.stem; directory.mkdir(exist_ok=True)
    latest=output.with_name(output.stem+'_last.pth')
    model=GPTModel(config).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.1)
    warmup=min(500,max(1,len(loader)//20))
    schedule=lambda step:min(1.,(step+1)/warmup)*(.1+.9*.5*(1+math.cos(math.pi*max(0,step-warmup)/max(1,len(loader)-warmup))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    amp=device.type=='cuda'
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    step=tokens=0
    elapsed=0.
    if latest.exists():
        saved=torch.load(latest,map_location=device,weights_only=True)
        if saved['config']!=config or saved['dataset']!=manifest or saved['target_tokens']!=budget or saved['batch_size']!=batch:
            raise ValueError('Resume settings differ.')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']); scaler.load_state_dict(saved['scaler'])
        step,tokens,elapsed=saved['step'],saved['tokens_processed'],saved['elapsed_seconds']
        torch.set_rng_state(saved['rng'].cpu()); torch.cuda.set_rng_state_all(saved['cuda_rng']) if amp else None
    started=time.monotonic()
    interrupted=False
    def stop(signum,frame):
        nonlocal interrupted
        interrupted=True
    for sig in (signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,stop)
    def checkpoint(path,metrics=None):
        save_checkpoint(path,model,config,dataset=manifest,target_tokens=budget,batch_size=batch,
                        step=step,tokens_processed=tokens,elapsed_seconds=elapsed+time.monotonic()-started,
                        optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),
                        rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if amp else [],metrics=metrics or {})
    def evaluate():
        model.eval(); total=count=0
        with torch.no_grad():
            for x,y in val_loader:
                x,y=x.to(device),y.to(device)
                with torch.amp.autocast('cuda', enabled=device.type=='cuda'):
                    loss=torch.nn.functional.cross_entropy(model(x).flatten(0,1),y.flatten(),reduction='sum')
                total+=loss.item(); count+=y.numel()
        model.train()
        return total/count
    write_json(directory/'config.json',dict(model=config,target_tokens=budget,batch_size=batch))
    write_json(directory/'dataset.json',manifest)
    print('Training parameters:',parameter_count(config),'| fresh target tokens:',len(dataset)*config['context_length'],flush=True)
    model.train()
    completed=True
    final={}
    for index,(x,y) in enumerate(loader):
        if index<step:
            continue
        if interrupted or elapsed+time.monotonic()-started>=max_hours*3600:
            completed=False; break
        x,y=x.to(device),y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=device.type=='cuda'):
            loss=torch.nn.functional.cross_entropy(model(x).flatten(0,1),y.flatten())
        if not torch.isfinite(loss):
            checkpoint(latest)
            raise RuntimeError('Nonfinite packed training loss.')
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        scaler.step(optimizer); scaler.update(); scheduler.step()
        step+=1; tokens+=y.numel()
        if step%100==0:
            print(json.dumps(dict(step=step,tokens=tokens,loss=loss.item(),elapsed_seconds=elapsed+time.monotonic()-started)),flush=True)
        if step%save_every==0:
            checkpoint(latest)
        if step%2000==0 or index+1==len(loader):
            val_loss=evaluate()
            final=dict(step=step,tokens_processed=tokens,validation_loss=val_loss,
                       validation_perplexity=math.exp(min(val_loss,80)),train_loss=loss.item())
            with (directory/'metrics.jsonl').open('a') as handle:
                handle.write(json.dumps(final)+'\n')
            write_json(directory/'metrics.json',final)
            checkpoint(latest,final)
    checkpoint(latest,final)
    if not completed:
        write_json(directory/'interrupted.json',dict(step=step,tokens_processed=tokens,state='resumable',max_hours=max_hours))
        raise SystemExit(124)
    if not final:
        final=json.loads((directory/'metrics.json').read_text())
    checkpoint(output,final)
    write_json(directory/'metrics.json',dict(final,parameters=parameter_count(config),tokens_per_parameter=tokens/parameter_count(config),completed=True))
    print('Completed fresh-token training:',output,flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='data/large')
    p.add_argument('--output',required=True)
    p.add_argument('--profile',default='runs/large/profile.json')
    p.add_argument('--benchmark',action='store_true')
    p.add_argument('--max-hours',type=float,default=20)
    a=p.parse_args()
    if a.benchmark:
        profile(a.output,max_hours=a.max_hours)
    else:
        train(a.data,a.output,a.profile,max_hours=a.max_hours)

if __name__=='__main__':
    main()
