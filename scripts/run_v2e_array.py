#!/usr/bin/env python3
"""Run the official v2e EventEmulator on an NPZ frame sequence."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--v2e-root",type=Path,required=True); p.add_argument("--seed",type=int,default=17); a=p.parse_args()
    sys.path.insert(0,str(a.v2e_root)); from v2ecore.emulator import EventEmulator
    with np.load(a.input) as d: frames=np.asarray(d["frames"],np.float32); timestamps=np.asarray(d["timestamps"],np.float64)
    h,w=frames.shape[1:]; emu=EventEmulator(pos_thres=.2,neg_thres=.2,sigma_thres=.03,cutoff_hz=0.,leak_rate_hz=.1,refractory_period_s=0.,shot_noise_rate_hz=0.,seed=a.seed,output_width=w,output_height=h,device="cpu"); chunks=[]
    for frame,t in zip(frames,timestamps):
        events=emu.generate_events(np.rint(np.clip(frame,0,1)*255).astype(np.uint8),float(t))
        if events is not None and len(events): chunks.append(np.asarray(events))
    emu.cleanup(); e=np.concatenate(chunks) if chunks else np.empty((0,4),np.float64); a.output.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(a.output,t=e[:,0].astype(np.float64),x=e[:,1].astype(np.int32),y=e[:,2].astype(np.int32),p=np.where(e[:,3]>0,1,-1).astype(np.int8)); return 0
if __name__=="__main__": raise SystemExit(main())
