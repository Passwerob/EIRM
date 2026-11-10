#!/usr/bin/env python3
"""Run the official DVS-Voltmeter EventSim on an NPZ frame sequence."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--dvs-root",type=Path,required=True); p.add_argument("--seed",type=int,default=17); a=p.parse_args(); sys.path.insert(0,str(a.dvs_root)); from src.config import cfg; from src.simulator import EventSim
    with np.load(a.input) as d: frames=np.asarray(d["frames"],np.float32); timestamps=np.asarray(d["timestamps"],np.float64)
    np.random.seed(a.seed); cfg.SENSOR.CAMERA_TYPE="DVS346"; cfg.SENSOR.K=[.00018*29250,20,.0001,1e-7,5e-9,.00001]; cfg.SENSOR.USE_GPU=False; cfg.SENSOR.DEVICE="cpu"; cfg.SENSOR.DTYPE="float32"; emu=EventSim(cfg=cfg,output_folder=".",video_name="array"); chunks=[]
    for frame,t in zip(frames,timestamps):
        events=emu.generate_events(np.rint(np.clip(frame,0,1)*255).astype(np.uint8),int(round(float(t)*1e6)))
        if events is not None and len(events): chunks.append(np.asarray(events))
    emu.reset(); e=np.concatenate(chunks) if chunks else np.empty((0,4),np.float64); a.output.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(a.output,t=e[:,0].astype(np.float64)*1e-6+float(timestamps[0]),x=e[:,1].astype(np.int32),y=e[:,2].astype(np.int32),p=np.where(e[:,3]>0,1,-1).astype(np.int8)); return 0
if __name__=="__main__": raise SystemExit(main())
