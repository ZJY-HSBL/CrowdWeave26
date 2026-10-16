"""Synthetic GOTP microbenchmark, NOT a detector accuracy or GPU training benchmark.

Benchmarks *only* unique-GT representative selection, pair penalty and its
backward, with warmup and synchronized device timings. The validation benchmark
does not use real CrowdHuman images or pretrained YOLO weights.
"""
from __future__ import annotations
import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
from crowdweave26.gotp import representative_pair_loss


def synthetic_batch(people: int, positives: int, device: torch.device):
    if people < 2 or positives < 1:
        raise ValueError("Require >=2 people and >=1 positive anchor each")
    grid = math.ceil(math.sqrt(people))
    ids = torch.arange(people, device=device)
    cx = (ids % grid).float() * 19.0 + 100.0
    cy = (ids // grid).float() * 34.0 + 100.0
    half_w, half_h = 13.0, 26.0
    gt_unique = torch.stack((cx-half_w,cy-half_h,cx+half_w,cy+half_h),-1)
    # Expand predictions to model excess overlap of neighboring persons.
    delta = torch.tensor([6., 5., 6., 5.], device=device)
    pred_unique = gt_unique + delta * torch.tensor([-1.,-1.,1.,1.],device=device)
    gt = gt_unique.repeat_interleave(positives, 0)[None]
    pred = pred_unique.repeat_interleave(positives, 0)[None]
    foreground = torch.ones((1,people*positives),device=device,dtype=torch.bool)
    index = ids.repeat_interleave(positives)[None]
    quality = torch.arange(positives,0,-1,device=device,dtype=torch.float32)
    quality = quality.repeat(people)[None,:,None] / positives
    return pred,gt,foreground,index,quality


def run_case(people, positives, repeats, warmup, device):
    original,gt,fg,idx,q = synthetic_batch(people,positives,device)
    samples=[]
    active=[]
    nonzero=[]
    losses=[]
    counts=[]
    for j in range(warmup+repeats):
        pred = original.detach().clone().requires_grad_(True)
        if device.type=="cuda":
            torch.cuda.synchronize(device)
        start=time.perf_counter()
        loss,stats=representative_pair_loss(
            pred,gt,fg,idx,q,margin=0.03,max_gt=128,max_pairs=4096,
            return_stats=True,
        )
        loss.backward()
        if device.type=="cuda":
            torch.cuda.synchronize(device)
        elapsed=(time.perf_counter()-start)*1000.0
        if j>=warmup:
            pair_count=int(stats[1].item())
            active_count=int(stats[2].item())
            samples.append(elapsed)
            active.append(active_count/max(1,pair_count))
            nonzero.append(int(bool((pred.grad.abs()>0).any())))
            losses.append(float(loss.detach().item()))
            counts.append(pair_count)
    ordered=sorted(samples)
    return {
        "people":people, "positives_per_person":positives,
        "total_positive_anchors":people*positives,
        "mean_step_ms":round(statistics.mean(samples),4),
        "median_step_ms":round(statistics.median(samples),4),
        "p95_step_ms":round(ordered[min(len(ordered)-1,math.ceil(0.95*len(ordered))-1)],4),
        "mean_close_pairs":round(statistics.mean(counts),3),
        "mean_violating_fraction":round(statistics.mean(active),6),
        "mean_raw_gotp_loss":round(statistics.mean(losses),6),
        "nonzero_gradient_step_fraction":round(statistics.mean(nonzero),6),
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--people",type=int,nargs="+",default=[8,16,32,64])
    parser.add_argument("--positives",type=int,default=3)
    parser.add_argument("--repeats",type=int,default=30)
    parser.add_argument("--warmup",type=int,default=5)
    parser.add_argument("--seed",type=int,default=2026)
    parser.add_argument("--device",default="cpu",choices=("cpu","cuda"))
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if min(args.people) < 2 or args.positives<1 or args.repeats<1 or args.warmup<0:
        parser.error("people >=2, positives >=1, repeats >=1, warmup >=0 required")
    if args.device=="cuda" and not torch.cuda.is_available():
        parser.error("CUDA device requested but not available")
    torch.manual_seed(args.seed)
    device=torch.device(args.device)
    results={
        "benchmark":"synthetic GOTP forward and backward microbenchmark only",
        "torch_version":torch.__version__,"device":str(device),
        "seed":args.seed,"repeats":args.repeats,"warmup":args.warmup,
        "results":[run_case(n,args.positives,args.repeats,args.warmup,device)
                   for n in args.people],
    }
    payload=json.dumps(results,indent=2)
    print(payload)
    if args.output is not None:
        if args.output.exists():
            parser.error("Output exists, refusing overwrite")
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(payload+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
