#!/usr/bin/env python3
"""Native-resolution diagnostic replay of a frozen 320px controller schedule.

This is a local warmed-up visualization, not a native-resolution closed-loop
evaluation and not an official DVS-Voltmeter/v2e baseline.
"""
import argparse
import json
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
import torch

from ami_simulator.events import ThresholdEventSimulator
from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.pipeline import rgb_to_linear_luminance
from ami_simulator.rotation import ConstantRotation
from ami_simulator.types import CameraIntrinsics, EventStream
from ami_simulator.vfi import PracticalRIFEBackend
from ami_simulator.warp import PrismWarper, compensate_events
from scripts.run_eveyes_v2_stateful import event_image


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--schedule',type=Path,required=True)
    p.add_argument('--source',type=Path,default=Path('data/eveyes_co_motion_v2/test/Scene_005/co_motion'))
    p.add_argument('--frame',type=int,default=180)
    p.add_argument('--warmup',type=int,default=15)
    args=p.parse_args()
    torch.set_num_threads(4); cv2.setNumThreads(1)
    with np.load(args.schedule) as z:
        rpm=z['rpm']; ts=z['timestamps']
    k=json.loads((args.source/'camera_intrinsics.json').read_text())
    camera=CameraIntrinsics(k['width'],k['height'],k['fx'],k['fy'],k['cx'],k['cy'])
    prism=RotatingWedgePrism(camera,.5)
    rife=PracticalRIFEBackend('/data/fcr/code/event_vggt/Data_Part/Practical-RIFE',device='cpu',inference_batch_size=3)
    start=args.frame-args.warmup
    if start<0 or args.frame>=len(rpm): raise ValueError('Invalid visualization window')
    # Integrate the entire recorded command history for exact actuator phase.
    phase=float(np.sum(rpm[:start]*2*np.pi/60*np.diff(ts[:start+1])))
    references=[None,None]
    simulator=ThresholdEventSimulator(.2)
    first=iio.imread(args.source/'rgb'/f'{start:06d}.png')
    for i in range(start,args.frame+1):
        second=iio.imread(args.source/'rgb'/f'{i+1:06d}.png')
        mid=rife.interpolate(first,second,np.arange(1,12)/12)
        frames=np.concatenate([first[None].astype(np.float32)/255,mid,second[None].astype(np.float32)/255])
        lum=rgb_to_linear_luminance(frames)
        times=np.linspace(ts[i],ts[i]+1/60,13)
        rotation=ConstantRotation(float(rpm[i]))
        warper=PrismWarper(prism,rotation,height=k['height'],width=k['width'],theta0=phase,reference_time=float(ts[i]))
        with torch.inference_mode(): warped,_=warper.warp(torch.from_numpy(lum).unsqueeze(1),times)
        passive,references[0]=simulator.simulate_with_state(lum,times,reference_log=references[0])
        active,references[1]=simulator.simulate_with_state(warped[:,0].numpy(),times,reference_log=references[1])
        if i==args.frame:
            mapped=compensate_events(active,prism,rotation,theta0=phase,reference_time=float(ts[i]))
            x=np.rint(mapped.extras['x_reference']); y=np.rint(mapped.extras['y_reference'])
            keep=(x>=0)&(x<k['width'])&(y>=0)&(y<k['height'])
            aligned=EventStream(x[keep],y[keep],active.t[keep],active.p[keep])
            panels=[]
            for name,im in [('Source RGB',first),('Passive threshold events',event_image(passive,k['height'],k['width'])),('Prism events: sensor coordinates',event_image(active,k['height'],k['width'])),('Prism events: optical alignment',event_image(aligned,k['height'],k['width']))]:
                tile=np.zeros((k['height']+40,k['width'],3),np.uint8); tile[40:]=im
                cv2.putText(tile,name,(12,27),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1)
                panels.append(tile)
            output=args.schedule.with_name(f'{args.schedule.stem}_native_frame{args.frame}.png')
            iio.imwrite(output,np.concatenate(panels,axis=1))
            output.with_suffix('.json').write_text(json.dumps({'source':str(args.source),'schedule':str(args.schedule),'source_resolution':[k['width'],k['height']],'frame':args.frame,'window_s':[float(times[0]),float(times[-1])],'warmup_s':args.warmup/60,'rpm':float(rpm[i]),'sampling_hz':720,'interpolation':'Practical-RIFE','caveat':'native replay of 320px closed-loop schedule; local warmup; not native closed-loop scores; passive threshold is not official DVS-Voltmeter or v2e'},indent=2)+'\n')
            print(output,flush=True)
        phase=(phase+rotation.omega_rad_s/60)%(2*np.pi)
        first=second
        print(f'native frame {i}/{args.frame}',flush=True)


if __name__=='__main__':
    main()
