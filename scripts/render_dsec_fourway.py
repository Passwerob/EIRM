#!/usr/bin/env python3
"""Generate one DSEC RGB/DVS/v2e/Ours qualitative four-way panel."""
from __future__ import annotations
import argparse, tempfile
from pathlib import Path
import cv2, imageio.v3 as iio, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.pipeline import rgb_to_linear_luminance
from ami_simulator.types import CameraIntrinsics
from scripts.evaluate_dvs_v2e_ours import _external, _slice
from scripts.evaluate_dl3dv_downstream import _ami_windows
from scripts.plot_eveyes_event_qualitative import _events_to_map
from ami_simulator.vfi import PracticalRIFEBackend

def _polarity_map(events, height, width):
 return _events_to_map(events, height, width, storm=False)

def main():
 p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path('/data/fcr/data/DSEC/thun_00_a_small')); p.add_argument('--frame',type=int,default=12); p.add_argument('--window-ms',type=float,default=10.0); p.add_argument('--prism-angle',type=float,default=0.1); p.add_argument('--rpm',type=float,default=60.0); p.add_argument('--contrast',type=float,default=0.5); p.add_argument('--output',type=Path,default=Path('results/ami_next/figures/dsec_fourway_10ms'))
 a=p.parse_args(); imgs=sorted((a.root/'images').glob('*.png')); frames=np.stack([iio.imread(x) for x in imgs]).astype(np.float32); source_times=np.loadtxt(a.root/'thun_00_a_image_timestamps.txt')[:len(frames)]*1e-6; source_times-=source_times[0]; frames=np.stack([cv2.resize(x,(320,240),interpolation=cv2.INTER_AREA) for x in frames]); h,w=frames.shape[1:3]
 i=min(a.frame,len(frames)-2); base0,base1=frames[i],frames[i+1]; source_dt=float(source_times[i+1]-source_times[i]); duration=min(float(a.window_ms)*1e-3,source_dt); rife=PracticalRIFEBackend('/data/fcr/code/event_vggt/Data_Part/Practical-RIFE',device='cpu',inference_batch_size=4); fraction=duration/source_dt; interp=rife.interpolate(base0,base1,np.asarray([fraction],dtype=np.float32)); short_frames=np.concatenate([base0[None],interp],axis=0); frames=short_frames; times=np.asarray([0.0,duration],dtype=np.float64); lum=rgb_to_linear_luminance(frames)
 warm_frames=np.concatenate([np.stack([cv2.resize(x,(320,240),interpolation=cv2.INTER_AREA) for x in np.stack([iio.imread(x) for x in imgs[:i+1]])]).astype(np.float32),short_frames[1:]],axis=0)
 warm_times=np.concatenate([source_times[:i+1]-source_times[i],np.asarray([duration])]); warm_lum=rgb_to_linear_luminance(warm_frames)
 ext=argparse.Namespace(dvs_python=Path('/data/fcr/miniconda3/envs/dvsvolt/bin/python'),dvs_root=Path('/data/fcr/code/event_vggt/Data_Part/DVS-Voltmeter'),v2e_python=Path('/data/fcr/miniconda3/envs/v2e/bin/python'),v2e_root=Path('/data/fcr/vendor/v2e'),seed=17)
 with tempfile.TemporaryDirectory(prefix='dsec_fourway_') as td: dvs=_external('dvs_voltmeter',lum,times,ext,Path(td)); v2e=_external('v2e',lum,times,ext,Path(td))
 k=CameraIntrinsics(width=320,height=240,fx=569.7633*0.5,fy=569.7633*0.5,cx=335.1*0.5,cy=221.24*0.5); prism=RotatingWedgePrism(k,a.prism_angle); masks=np.ones((len(warm_lum),h,w),bool); ref,_,_=_ami_windows(warm_lum,warm_times,masks,prism,a.rpm,device='cpu',contrast_threshold=a.contrast); ev=[_slice(x,0.0,duration) for x in (dvs,v2e)]; ours=ref[-1]
 print({'window_ms':duration*1000.0,'dvs_events':len(ev[0]),'v2e_events':len(ev[1]),'ours_events':len(ours.x)},flush=True)
 panels=[frames[0].astype(np.uint8),_polarity_map(ev[0],h,w),_polarity_map(ev[1],h,w),_polarity_map(ours,h,w)]; labels=['RGB','DVS-Voltmeter','v2e','Ours (180 RPM, inverse-warp)']; fig,ax=plt.subplots(1,4,figsize=(16,4),dpi=160)
 for q,im,l in zip(ax,panels,labels): q.imshow(im); q.set_title(l); q.axis('off')
 fig.suptitle(f'DSEC thun_00_a · frame {i}→{i+1} · reconstructed {duration*1000:.1f} ms window'); a.output.parent.mkdir(parents=True,exist_ok=True); fig.savefig(a.output.with_suffix('.png'),bbox_inches='tight'); fig.savefig(a.output.with_suffix('.pdf'),bbox_inches='tight'); a.output.with_suffix('.json').write_text(__import__('json').dumps({'dataset':'DSEC','scene':'thun_00_a','source_pair':[i,i+1],'source_interval_ms':source_dt*1000,'window_ms':duration*1000,'temporal_reconstruction':'Practical-RIFE single intermediate frame','event_generators':['DVS-Voltmeter','v2e','Ours'],'ours':'RPM '+str(a.rpm)+' inverse-warped','prism_angle_deg':a.prism_angle},indent=2)+'\n'); print(a.output.with_suffix('.png'))
if __name__=='__main__': main()
