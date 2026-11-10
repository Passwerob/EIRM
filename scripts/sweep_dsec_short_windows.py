#!/usr/bin/env python3
"""DSEC short-window sweep on one RIFE-densified RGB clip."""
from __future__ import annotations
import argparse, json, tempfile
from pathlib import Path
import cv2, imageio.v3 as iio, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.pipeline import rgb_to_linear_luminance
from ami_simulator.types import CameraIntrinsics
from ami_simulator.vfi import PracticalRIFEBackend
from scripts.evaluate_dvs_v2e_ours import _external, _slice
from scripts.evaluate_dl3dv_downstream import _ami_windows
from scripts.render_eveyes_scene001_storm_vs_ours import _concat

def polarity_map(events, height, width, max_points=8000):
    if len(events) > max_points:
        rng=np.random.default_rng(17)
        keep=rng.choice(len(events),size=max_points,replace=False)
        events=type(events)(events.x[keep],events.y[keep],events.t[keep],events.p[keep])
    pos=np.zeros((height,width),np.float32); neg=np.zeros((height,width),np.float32)
    if len(events):
        x=np.rint(events.x).astype(int); y=np.rint(events.y).astype(int); ok=(x>=0)&(x<width)&(y>=0)&(y<height)
        np.add.at(pos,(y[ok&(events.p>0)],x[ok&(events.p>0)]),1); np.add.at(neg,(y[ok&(events.p<0)],x[ok&(events.p<0)]),1)
    vals=np.concatenate([pos[pos>0],neg[neg>0]]) if np.any(pos>0)|np.any(neg>0) else np.ones(1)
    peak=max(float(np.percentile(vals,99.5)),1.)
    pos=np.clip(np.log1p(pos)/np.log1p(peak),0,1); neg=np.clip(np.log1p(neg)/np.log1p(peak),0,1)
    out=np.zeros((height,width,3),np.uint8); out[...,0]=(255*pos).astype(np.uint8); out[...,2]=(255*neg).astype(np.uint8); return out

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path('/data/fcr/data/DSEC/thun_00_a_small')); p.add_argument('--start-frame',type=int,default=12); p.add_argument('--output',type=Path,default=Path('results/ami_next/figures/dsec_window_sweep')); p.add_argument('--rpm',type=float,default=180.); p.add_argument('--prism-angle',type=float,default=.1); p.add_argument('--contrast',type=float,default=.5); p.add_argument('--max-ms',type=float,default=200.)
    a=p.parse_args(); imgs=sorted((a.root/'images').glob('*.png')); raw=np.stack([iio.imread(x) for x in imgs]).astype(np.float32); source=np.loadtxt(a.root/'thun_00_a_image_timestamps.txt')[:len(raw)]*1e-6; source-=source[0]; raw=np.stack([cv2.resize(x,(320,240),interpolation=cv2.INTER_AREA) for x in raw]); i=a.start_frame; dt=float(source[i+1]-source[i]); needed=int(np.ceil(a.max_ms/(dt*1000)))+1; base=raw[i:i+needed+1]; rife=PracticalRIFEBackend('/data/fcr/code/event_vggt/Data_Part/Practical-RIFE',device='cpu',inference_batch_size=4)
    frames=[]; times=[]; step=dt/5
    for j in range(len(base)-1):
        mids=rife.interpolate(base[j],base[j+1],np.arange(1,5,dtype=np.float32)/5); frames.extend([base[j],*mids]); times.extend([j*dt+k*step for k in range(5)])
    frames.append(base[-1]); times.append((len(base)-1)*dt); frames=np.asarray(frames); times=np.asarray(times); max_t=min(a.max_ms/1000.,float(times[-1])); keep=times<=max_t+1e-9; frames=frames[keep]; times=times[keep]; lum=rgb_to_linear_luminance(frames); h,w=lum.shape[1:]
    ext=argparse.Namespace(dvs_python=Path('/data/fcr/miniconda3/envs/dvsvolt/bin/python'),dvs_root=Path('/data/fcr/code/event_vggt/Data_Part/DVS-Voltmeter'),v2e_python=Path('/data/fcr/miniconda3/envs/v2e/bin/python'),v2e_root=Path('/data/fcr/vendor/v2e'),seed=17)
    with tempfile.TemporaryDirectory(prefix='dsec_sweep_') as td: dvs=_external('dvs_voltmeter',lum,times,ext,Path(td)); v2e=_external('v2e',lum,times,ext,Path(td))
    k=CameraIntrinsics(width=w,height=h,fx=569.7633*.5,fy=569.7633*.5,cx=335.1*.5,cy=221.24*.5); prism=RotatingWedgePrism(k,a.prism_angle); ref,_,_=_ami_windows(lum,times,np.ones((len(lum),h,w),bool),prism,a.rpm,device='cpu',contrast_threshold=a.contrast)
    cuts=[x/1000 for x in (50,100,150,200,250) if x<=a.max_ms+1e-9]; a.output.parent.mkdir(parents=True,exist_ok=True); rows=[]; rgb=frames[0].astype(np.uint8)
    fig,ax=plt.subplots(len(cuts),4,figsize=(16,3.2*len(cuts)),dpi=140,squeeze=False)
    for row,cut in enumerate(cuts):
        idx=max(1,int(round(cut/step))); end=float(times[min(idx,len(times)-1)]); streams=[_slice(dvs,0,end),_slice(v2e,0,end),_concat(ref[:min(idx,len(ref))])]; counts=[len(x) for x in streams]
        # Keep the quantitative count cumulative, but render an instantaneous tail
        # so long windows do not collapse into an unreadable full-frame carpet.
        tail_ms=30.0; tail_start=max(0.0,end-tail_ms/1000.0); display_streams=[_slice(dvs,tail_start,end),_slice(v2e,tail_start,end),_concat(ref[max(0,idx-3):min(idx,len(ref))])]
        rows.append({'window_ms':cut*1000,'dvs_events':counts[0],'v2e_events':counts[1],'ours_events':counts[2],'display_window_ms':tail_ms})
        raw_idx=min(i+max(1,int(round(cut/dt))),len(raw)-1)
        panels=[raw[raw_idx].astype(np.uint8)]+[polarity_map(x,h,w) for x in display_streams]; labels=['RGB at window end',f'DVS-Voltmeter ({counts[0]:,} cumulative; display <=8k)',f'v2e ({counts[1]:,} cumulative; display <=8k)',f'Ours ({counts[2]:,} cumulative; display <=8k)']
        for col,(image,label) in enumerate(zip(panels,labels)): ax[row,col].imshow(image); ax[row,col].set_title(f'{label} · {cut*1000:.0f} ms' if row==0 else label); ax[row,col].axis('off')
    fig.suptitle(f'DSEC thun_00_a · short-window sweep · {a.rpm:g} RPM · angle {a.prism_angle:g}° · contrast {a.contrast:g}'); fig.tight_layout(); fig.savefig(a.output.with_suffix('.png'),bbox_inches='tight'); fig.savefig(a.output.with_suffix('.pdf'),bbox_inches='tight'); a.output.with_suffix('.json').write_text(json.dumps({'dataset':'DSEC','scene':'thun_00_a','start_frame':i,'internal_step_ms':step*1000,'prism_angle_deg':a.prism_angle,'contrast_threshold':a.contrast,'rpm':a.rpm,'rows':rows,'temporal_reconstruction':'Practical-RIFE 5x densification between DSEC RGB frames'},indent=2)+'\n'); print(json.dumps(rows,indent=2))
if __name__=='__main__': main()
