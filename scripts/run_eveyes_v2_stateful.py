#!/usr/bin/env python3
"""Reproducible EVEYES v2 sensing pilot: RIFE, stateful teacher, causal GRU.

RGB/IMU are read by named seconds columns. No pose, motion-state GT, masks,
future flow, or edge scores enter the student. The teacher alone uses edge F1.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
import torch

from ami_simulator.dsec import read_imu_csv
from ami_simulator.event_quality import compute_reference_edge_mask, event_quality_metrics
from ami_simulator.events import ThresholdEventSimulator
from ami_simulator.flow import estimate_dense_flow, summarize_flow
from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.motion_net import CausalAnchorQGRUPolicy
from ami_simulator.pipeline import rgb_to_linear_luminance
from ami_simulator.rotation import ConstantRotation
from ami_simulator.types import CameraIntrinsics, EventStream
from ami_simulator.vfi import PracticalRIFEBackend
from ami_simulator.warp import PrismWarper, compensate_events

ACTIONS = (0., 180., 360.)
SPECS = {'train': ('train', 'Scene_002'), 'valid': ('train', 'Scene_004'), 'test': ('test', 'Scene_005')}
FEATURES = ([f'flow_{i}' for i in range(8)] +
            [f'{sensor}_{stat}_{axis}' for sensor in ('gyro', 'accel') for stat in ('mean', 'std') for axis in 'xyz'] +
            ['phase_sin', 'phase_cos', 'previous_rpm_over_360', 'dwell_s', 'window_s',
             'previous_event_density', 'previous_active_fraction', 'previous_polarity_imbalance'])


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def source(args, split):
    folder, scene = SPECS[split]
    return args.source / folder / scene / 'co_motion'


def prepare(args, split):
    root = source(args, split)
    dest = args.output / split
    dest.mkdir(parents=True, exist_ok=True)
    with (root / 'timestamps.csv').open() as f:
        rows = list(csv.DictReader(f))
    ts = np.array([float(r['timestamp_s']) for r in rows])
    assert len(ts) == 480 and np.allclose(np.diff(ts), 1/60, atol=1e-9)
    calibration = json.loads((root / 'camera_intrinsics.json').read_text())
    h = round(calibration['height'] * args.width / calibration['width'])
    rgb = np.stack([cv2.resize(iio.imread(root / 'rgb' / f'{i:06d}.png'), (args.width, h), interpolation=cv2.INTER_AREA) for i in range(len(ts))])
    imu = read_imu_csv(root / 'imu.csv', timestamp_unit='s')
    # Feature at i uses only the completed interval ending at timestamp i.
    obs = np.zeros((len(ts)-1, 20), np.float32)
    for i in range(1, len(obs)):
        obs[i, :8] = summarize_flow(estimate_dense_flow(rgb[i-1], rgb[i])) / (ts[i]-ts[i-1])
        keep = (imu.t > ts[i-1]) & (imu.t <= ts[i])
        assert keep.any()
        obs[i, 8:] = np.concatenate([v for a in (imu.gyro[keep], imu.accel[keep]) for v in (a.mean(0), a.std(0))])
    # Four subintervals per frame: 240 Hz, 9 degrees at 360 RPM.
    steps = args.substeps
    shape = (len(ts)-1, steps+1, h, args.width)
    lum_path = dest / 'luminance.npy'
    if not lum_path.exists():
        temp = dest / 'luminance.partial.npy'
        lum = np.lib.format.open_memmap(temp, mode='w+', dtype=np.float32, shape=shape)
        rife = PracticalRIFEBackend(args.rife, device='cpu', inference_batch_size=steps-1)
        for i in range(len(ts)-1):
            mid = rife.interpolate(rgb[i], rgb[i+1], np.arange(1,steps)/steps)
            values = np.concatenate([rgb[i:i+1].astype(np.float32)/255, mid, rgb[i+1:i+2].astype(np.float32)/255])
            lum[i] = rgb_to_linear_luminance(values)
            if i % 40 == 0:
                print(f'{split} RIFE {i+1}/{len(ts)-1}', flush=True)
        lum.flush()
        del lum
        temp.rename(lum_path)
    np.savez_compressed(dest / 'input.npz', timestamps=ts, observations=obs, rgb=rgb)
    dump(dest / 'source.json', {'source': str(root), 'timestamp_unit': 's', 'duration_s': float(ts[-1]-ts[0]), 'calibration': calibration, 'width': args.width, 'height': h, 'substeps': steps, 'interpolation': 'Practical-RIFE', 'feature_names': FEATURES, 'input_excludes': ['motion_state', 'pose', 'GT mask', 'GT edge', 'future observations']})
    print(f'{split} prepared {len(ts)} frames, {ts[-1]:.6f}s', flush=True)


class Scene:
    def __init__(self, args, split):
        self.args = args
        self.split = split
        root = args.output / split
        self.meta = json.loads((root / 'source.json').read_text())
        if self.meta['substeps'] != args.substeps or self.meta['width'] != args.width:
            raise ValueError('Cache configuration mismatch; use a separate output directory.')
        with np.load(root / 'input.npz') as z:
            self.ts, self.obs, self.rgb = z['timestamps'], z['observations'], z['rgb']
        self.lum = np.load(root / 'luminance.npy', mmap_mode='r')
        self.h, self.w = self.lum.shape[-2:]
        k = self.meta['calibration']
        self.k = CameraIntrinsics(k['width'], k['height'], k['fx'], k['fy'], k['cx'], k['cy']).resized(self.w, self.h)
        self.prism = RotatingWedgePrism(self.k, args.prism_angle)
        # Conservative shared valid crop for every method and prism phase.
        # Bound ray displacement at a grid of points, adding two pixel margin.
        yy, xx = np.mgrid[0:self.h:complex(0,17), 0:self.w:complex(0,17)]
        points = torch.tensor(np.stack([xx, yy], -1), dtype=torch.float32)
        displacement = max(float((self.prism.forward_points(points, float(t))-points).abs().max()) for t in np.linspace(0, 2*np.pi, 65))
        self.border = int(np.ceil(displacement))+2
        self.edges = []
        for i in range(len(self.ts)-1):
            edge = compute_reference_edge_mask(self.lum[i,0])
            edge[:self.border] = False; edge[-self.border:] = False
            edge[:,:self.border] = False; edge[:,-self.border:] = False
            self.edges.append(edge)

    def initial(self, rpm=180.):
        return {'phase': 0., 'reference': None, 'rpm': rpm, 'switch': -1., 'stats': np.zeros(3)}

    def feasible(self, state, i):
        if self.ts[i]-state['switch']+1e-10 < self.args.dwell:
            return (ACTIONS.index(state['rpm']),)
        return tuple(range(len(ACTIONS)))

    def feature(self, state, i):
        return np.concatenate([self.obs[i], [np.sin(state['phase']), np.cos(state['phase']), state['rpm']/360,
                               min(self.ts[i]-state['switch'],1.), self.ts[i+1]-self.ts[i]], state['stats']]).astype(np.float32)

    def step(self, state, i, rpm, passive=False):
        start, end = self.ts[i:i+2]
        times = np.linspace(start, end, self.lum.shape[1])
        rotation = ConstantRotation(rpm)
        if passive:
            sensor = self.lum[i]
        else:
            warper = PrismWarper(self.prism, rotation, height=self.h, width=self.w, theta0=state['phase'], reference_time=float(start))
            with torch.inference_mode():
                warped, _ = warper.warp(torch.from_numpy(np.array(self.lum[i])).unsqueeze(1), times)
            sensor = warped[:,0].numpy()
        raw, ref = ThresholdEventSimulator(.2).simulate_with_state(sensor, times, reference_log=state['reference'])
        # Ignore all sensor pixels that can observe padded content.
        b = self.border
        keep = (raw.x >= b)&(raw.x < self.w-b)&(raw.y >= b)&(raw.y < self.h-b)
        raw = EventStream(raw.x[keep], raw.y[keep], raw.t[keep], raw.p[keep])
        if passive:
            ev = raw
        else:
            mapped = compensate_events(raw, self.prism, rotation, theta0=state['phase'], reference_time=float(start))
            ev = EventStream(np.rint(mapped.extras['x_reference']), np.rint(mapped.extras['y_reference']), raw.t, raw.p)
        keep = (ev.x >= b)&(ev.x < self.w-b)&(ev.y >= b)&(ev.y < self.h-b)
        ev = EventStream(ev.x[keep], ev.y[keep], ev.t[keep], ev.p[keep])
        metrics = event_quality_metrics(ev, reference_edge_mask=self.edges[i], image_shape=(self.h,self.w))
        stats = np.array([metrics['event_density'], metrics['unique_pixel_ratio'], metrics['polarity_imbalance']])
        nxt = {'phase': float((state['phase']+rotation.omega_rad_s*(end-start))%(2*np.pi)), 'reference': ref,
               'rpm': rpm, 'switch': float(start) if rpm != state['rpm'] else state['switch'], 'stats': stats}
        return nxt, metrics, ev


def teacher(scene, state, i):
    q = np.zeros(len(ACTIONS), np.float32)
    results = {}
    for a in scene.feasible(state, i):
        nxt, m, ev = scene.step(state,i,ACTIONS[a])
        score = m['edge_f1']
        if i+1 < len(scene.ts)-1:
            second = [scene.step(nxt,i+1,ACTIONS[b])[1]['edge_f1'] for b in scene.feasible(nxt,i+1)]
            score = (score+max(second))/2
        q[a] = score
        results[a] = (nxt,m,ev)
    return q, results


def windows(features, length):
    a = np.asarray(features, np.float32)
    return np.stack([np.concatenate([np.zeros((max(0,length-i-1),a.shape[1]),np.float32), a[max(0,i-length+1):i+1]]) for i in range(len(a))])


def load_policy(path):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    model = CausalAnchorQGRUPolicy(len(FEATURES), 64, len(ACTIONS))
    model.load_state_dict(ck['model']); model.eval()
    return model, ck


def choose(model, ck, history, feasible):
    values = windows(history[-ck['window']:], ck['window'])[-1:]
    values = np.clip((values-np.array(ck['mean']))/np.array(ck['std']),-5,5).astype(np.float32)
    with torch.inference_mode():
        q = model(torch.from_numpy(values))[0].numpy()
    return max(feasible, key=lambda a: float(q[a]))


def collect(args, split):
    scene = Scene(args,split)
    model_ck = load_policy(args.policy) if args.policy else None
    state = scene.initial(); features=[]; targets=[]; allowed=[]; executed=[]
    for i in range(len(scene.ts)-1):
        features.append(scene.feature(state,i))
        q, results = teacher(scene,state,i)
        mask = np.zeros(len(ACTIONS),bool); mask[list(results)] = True
        targets.append(q); allowed.append(mask)
        # Student-state collection can be used for a train-only DAgger pass.
        a = choose(*model_ck, features, tuple(results)) if model_ck else max(results, key=lambda a: (q[a], -abs(ACTIONS[a]-state['rpm'])))
        state = results[a][0]; executed.append(a)
        if i%40==0:
            print(f'{split} teacher {i+1}/479 actions={np.bincount(executed,minlength=3).tolist()}',flush=True)
    suffix = '_dagger' if args.policy else ''
    np.savez_compressed(args.output / split / f'teacher{suffix}.npz', features=features, q=targets, feasible=allowed, executed=executed)


def fit(args):
    def read(split):
        xs=[]; ys=[]; ms=[]
        paths = [args.output/split/'teacher.npz']
        if split=='train' and (args.output/split/'teacher_dagger.npz').exists():
            paths.append(args.output/split/'teacher_dagger.npz')
        for p in paths:
            with np.load(p) as z:
                xs.append(windows(z['features'],8)); ys.append(z['q']); ms.append(z['feasible'])
        return np.concatenate(xs), np.concatenate(ys), np.concatenate(ms)
    tx,ty,tm = read('train'); vx,vy,vm = read('valid')
    mean=tx.mean((0,1)); std=tx.std((0,1)); std[std<1e-5]=1
    tx=torch.tensor(np.clip((tx-mean)/std,-5,5)); vx=torch.tensor(np.clip((vx-mean)/std,-5,5))
    ty=torch.tensor(ty); vy=torch.tensor(vy); tm=torch.tensor(tm); vm=torch.tensor(vm)
    torch.manual_seed(args.seed)
    model=CausalAnchorQGRUPolicy(len(FEATURES),64,len(ACTIONS))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    def loss(pred,y,mask):
        # Only feasible decisions train the ranking; no penalties on event counts.
        valid=mask.sum(1)>1
        pred=pred[valid]; y=y[valid]; mask=mask[valid]
        pred=pred-pred.mean(1,keepdim=True); y=y-y.mean(1,keepdim=True)
        regression=torch.nn.functional.mse_loss(pred[mask],y[mask])
        target=torch.softmax(y.masked_fill(~mask,-1e4)/.05,dim=1)
        rank=-(target*torch.log_softmax(pred.masked_fill(~mask,-1e4)/.05,dim=1)).sum(1).mean()
        return regression+.02*rank
    best=float('inf'); logs=[]; best_state=None; best_epoch=0
    for epoch in range(args.epochs):
        model.train(); optimizer.zero_grad(); l=loss(model(tx),ty,tm); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5); optimizer.step()
        if epoch%10==0 or epoch==args.epochs-1:
            model.eval()
            with torch.inference_mode(): v=float(loss(model(vx),vy,vm))
            logs.append({'epoch':epoch,'train_loss':float(l),'validation_loss':v})
            if v<best:
                best=v; best_epoch=epoch; best_state={k:t.detach().clone() for k,t in model.state_dict().items()}
    checkpoint={'model':best_state,'mean':mean.tolist(),'std':std.tolist(),'window':8,'feature_names':FEATURES,'actions':list(ACTIONS),'epoch':best_epoch,'seed':args.seed,'teacher':'stateful_horizon2_edge_f1','interpolation':'practical_rife','timestamp_unit':'s'}
    target=args.output/f'policy_seed{args.seed}.pt'
    torch.save(checkpoint,target)
    dump(target.with_suffix('.json'),{'best_epoch':best_epoch,'validation_loss':best,'training_windows':len(tx),'validation_windows':len(vx),'history':logs,'features':FEATURES,'evaluation_required':'closed_loop_rollout'})
    print(f'Trained {target}: epoch {best_epoch}, validation loss {best:.6f}',flush=True)


def event_image(events,h,w):
    im=np.zeros((h,w,3),np.uint8)
    for positive,color in ((True,(255,70,70)),(False,(30,220,255))):
        m=(events.p>0)==positive
        im[events.y[m],events.x[m]]=color
    return im


def evaluate(args, split):
    scene=Scene(args,split)
    model,ck=load_policy(args.policy)
    methods=['no_prism','fixed_180','fixed_360','learned']
    report={}; selected_panels={}; learned_schedule=[]
    for method in methods:
        rpm=360. if method=='fixed_360' else 180.
        if method=='no_prism': rpm=0.
        state=scene.initial(rpm); history=[]; rows=[]
        for i in range(len(scene.ts)-1):
            history.append(scene.feature(state,i))
            if method=='learned': rpm=ACTIONS[choose(model,ck,history,scene.feasible(state,i))]
            state,metrics,ev=scene.step(state,i,rpm,passive=method=='no_prism')
            rows.append({'t':float(scene.ts[i]),'rpm':rpm,**metrics})
            if method=='learned': learned_schedule.append(rpm)
            if i in (60,180,300,420): selected_panels[method,i]=event_image(ev,scene.h,scene.w)
            if i%120==0: print(f'{split} {method} {i+1}/479',flush=True)
        keys=('edge_f1','edge_precision','edge_coverage','event_count')
        report[method]={'mean':{k:float(np.mean([r[k] for r in rows])) for k in keys},
                        'worst_1s_edge_f1':min(float(np.mean([r['edge_f1'] for r in rows[j:j+60]])) for j in range(0,len(rows),60)),
                        'rpm_counts':{str(r):sum(x['rpm']==r for x in rows) for r in set(x['rpm'] for x in rows)},'rows':rows}
    out=args.output/f'{split}_{args.policy.stem}'
    dump(out.with_suffix('.json'),{'methods':report,'policy':str(args.policy),'split':split,'duration_s':float(scene.ts[-1]),'resolution':[scene.w,scene.h],
         'metric':'full-frame source-RGB edge support in shared valid crop; not target-specific GT',
         'threshold_state':'continuous','phase':'continuous','interpolation':'Practical-RIFE','event_count_is_not_utility':True,
         'scope':'one train source city, one test source city; validation shares training source city'})
    panels=[]
    for i in (60,180,300,420):
        row=[]
        for label,im in [('RGB',scene.rgb[i])]+[(m,selected_panels[m,i]) for m in methods]:
            tile=np.zeros((scene.h+28,scene.w,3),np.uint8); tile[28:]=im
            cv2.putText(tile,f'{label} t={scene.ts[i]:.2f}s',(5,19),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
            row.append(tile)
        panels.append(np.concatenate(row,axis=1))
    iio.imwrite(out.with_suffix('.png'),np.concatenate(panels,axis=0))
    np.savez_compressed(out.with_suffix('.npz'),timestamps=scene.ts[:-1],rpm=learned_schedule)
    print(json.dumps({m:r['mean'] for m,r in report.items()},indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','collect','fit','evaluate'])
    p.add_argument('--split',choices=list(SPECS),default='train')
    p.add_argument('--source',type=Path,default=Path('data/eveyes_co_motion_v2'))
    p.add_argument('--output',type=Path,default=Path('results/eveyes_v2_stateful_rife'))
    p.add_argument('--rife',type=Path,default=Path('/data/fcr/code/event_vggt/Data_Part/Practical-RIFE'))
    p.add_argument('--width',type=int,default=320)
    p.add_argument('--substeps',type=int,default=4)
    p.add_argument('--prism-angle',type=float,default=.5)
    p.add_argument('--dwell',type=float,default=.025)
    p.add_argument('--epochs',type=int,default=300)
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--policy',type=Path)
    args=p.parse_args()
    torch.set_num_threads(4); cv2.setNumThreads(1)
    args.output.mkdir(parents=True,exist_ok=True)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    dump(args.output/f'command_{args.stage}_{args.split}_{args.seed}.json',config)
    if args.stage=='prepare': prepare(args,args.split)
    elif args.stage=='collect': collect(args,args.split)
    elif args.stage=='fit': fit(args)
    else: evaluate(args,args.split)


if __name__=='__main__':
    main()
