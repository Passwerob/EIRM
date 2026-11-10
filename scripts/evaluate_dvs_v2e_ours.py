#!/usr/bin/env python3
"""Fair downstream comparison of DVS-Voltmeter, v2e, and learned AMI."""
from __future__ import annotations
import argparse,json,subprocess,tempfile
from pathlib import Path
import numpy as np
import torch
from ami_simulator.event_quality import compute_reference_edge_mask,event_quality_metrics
from ami_simulator.flow import estimate_dense_flow,summarize_motion_clusters
from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.motion_net import CausalAnchorQGRUPolicy
from ami_simulator.types import EventStream
from scripts.evaluate_anchor_feature_matching import _feature_metrics
from scripts.evaluate_anchor_residual_policy import CLUSTER_FEATURE_DIM,_state
from scripts.evaluate_phase_continuous_control import _load_scene
from scripts.run_control_headroom_stateful import RolloutState,_transition

Q_ACTIONS=(0.,180.,360.)

def _slice(events,t0,t1):
    keep=(events.t>t0)&(events.t<=t1); return EventStream(events.x[keep],events.y[keep],events.t[keep],events.p[keep])

def _read_npz(path):
    with np.load(path) as d: return EventStream(d["x"],d["y"],d["t"],d["p"]).sorted_by_time()

def _window_metrics(events,reference,mask,previous_active=None,bins=4):
    edge=compute_reference_edge_mask(reference,edge_quantile=.75)&mask; quality=event_quality_metrics(events,reference_edge_mask=edge,previous_active_pixels=previous_active,image_shape=reference.shape); feature=_feature_metrics(reference,events)
    if len(events):
        valid=(events.x>=0)&(events.x<reference.shape[1])&(events.y>=0)&(events.y<reference.shape[0]); on=valid.copy(); on[valid]&=edge[events.y[valid],events.x[valid]]; span=max(float(events.t.max()-events.t.min()),1e-12); idx=np.minimum(((events.t[on]-events.t.min())/span*bins).astype(int),bins-1); temporal=float(len(np.unique(idx))/bins) if len(idx) else 0.
    else: temporal=0.
    active=np.unique(events.y.astype(np.int64)*reference.shape[1]+events.x.astype(np.int64)) if len(events) else np.empty(0,np.int64)
    return {"edge_f1":float(quality["edge_f1"]),"edge_precision":float(quality["edge_precision"]),"edge_recall":float(quality["edge_coverage"]),"event_count":len(events),"temporal_support":temporal,"spatial_matching_f1":float(feature["spatial_f1"]),"feature_repeatability":float(feature["spatial_repeatability"]),"descriptor_true_matches":float(feature["descriptor_true_matches"]),"rgb_keypoints":int(feature["rgb_keypoints"]),"event_keypoints":int(feature["event_keypoints"]),"spatial_precision":float(feature["spatial_precision"]),"spatial_recall":float(feature["spatial_recall"])},active

def _external(method,frames,timestamps,args,tmp):
    input_path=tmp/"input.npz"; output=tmp/f"{method}.npz"; np.savez_compressed(input_path,frames=frames,timestamps=timestamps)
    if method=="dvs_voltmeter": cmd=[str(args.dvs_python),str(Path(__file__).with_name("run_dvs_voltmeter_array.py")),"--input",str(input_path),"--output",str(output),"--dvs-root",str(args.dvs_root),"--seed",str(args.seed)]
    else: cmd=[str(args.v2e_python),str(Path(__file__).with_name("run_v2e_array.py")),"--input",str(input_path),"--output",str(output),"--v2e-root",str(args.v2e_root),"--seed",str(args.seed)]
    subprocess.run(cmd,check=True); return _read_npz(output)

def _ours(scene,oracle,objective,model,ck,margin,confidence,dwell):
    lum,ts,masks,intrinsics=scene; prism=RotatingWedgePrism(intrinsics,2.); cluster=np.stack([summarize_motion_clusters(estimate_dense_flow(lum[i],lum[i+1]),clusters=3) for i in range(len(ts)-1)]).astype(np.float32); state=RolloutState(0.,None,None,180.,float(ts[0])-dwell); history=[]; streams=[]; mean=np.asarray(ck["feature_mean"],np.float32); std=np.asarray(ck["feature_std"],np.float32); window=int(ck["window"])
    for i in range(len(ts)-1):
        f=_state(oracle,index=i,previous=state.previous_metrics,phase=state.phase,dwell_seconds=float(ts[i]-state.last_switch_time),previous_rpm=state.previous_rpm,cluster_features=cluster); history.append(f); rpm=180.
        if i>=window-1:
            x=np.clip((np.stack(history[-window:])-mean)/std,-float(ck.get("feature_clip",5.)),float(ck.get("feature_clip",5.)))
            with torch.inference_mode(): q=model(torch.from_numpy(x[None].astype(np.float32)))[0].numpy()
            best=int(np.argmax(q)); prob=float(torch.softmax(torch.from_numpy(q),0)[best]); ready=float(ts[i]-state.last_switch_time)>=dwell
            if best!=1 and float(q[best]-q[1])>margin and prob>confidence and ready: rpm=Q_ACTIONS[best]
        result,state=_transition(scene,oracle,objective,prism,state,i,rpm,dwell=dwell); streams.append(result["reference_events"])
    return streams

def _aggregate(rows):
    methods=sorted(set(r["method"] for r in rows)); keys=("edge_f1","edge_precision","edge_recall","event_count","temporal_support","spatial_matching_f1","feature_repeatability","descriptor_true_matches")
    return {m:{k:float(np.mean([r[k] for r in rows if r["method"]==m])) for k in keys} for m in methods}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--manifest",type=Path,default=Path("results/comotion_v2_learned/manifest.json")); p.add_argument("--dataset-root",type=Path,default=Path("data/comotion_controlled_v2")); p.add_argument("--student",type=Path,default=Path("results/comotion_horizon2_q_policy/policy.pt")); p.add_argument("--split",choices=("valid","test"),default="valid"); p.add_argument("--output",type=Path,required=True); p.add_argument("--v2e-root",type=Path,default=Path("/data/fcr/vendor/v2e")); p.add_argument("--v2e-python",type=Path,default=Path("/data/fcr/miniconda3/envs/v2e/bin/python")); p.add_argument("--dvs-root",type=Path,default=Path("/data/fcr/code/event_vggt/Data_Part/DVS-Voltmeter")); p.add_argument("--dvs-python",type=Path,default=Path("/data/fcr/miniconda3/envs/dvsvolt/bin/python")); p.add_argument("--margin",type=float,default=.005); p.add_argument("--confidence",type=float,default=.35); p.add_argument("--dwell",type=float,default=.025); p.add_argument("--seed",type=int,default=17); p.add_argument("--max-scenes",type=int,default=0); a=p.parse_args()
    manifest=json.loads(a.manifest.read_text()); ck=torch.load(a.student,map_location="cpu",weights_only=True); model=CausalAnchorQGRUPolicy(CLUSTER_FEATURE_DIM,int(ck["hidden_dim"]),int(ck["num_actions"])); model.load_state_dict(ck["model"]); model.eval(); objective={**manifest["configuration"],"event_efficiency_weight":0.}; rows=[]
    completed=0
    for rec in manifest["oracles"]:
        if rec["split"]!=a.split: continue
        if a.max_scenes and completed>=a.max_scenes: break
        lum,ts,masks,intrinsics,oracle=_load_scene(a.dataset_root/str(rec["scene"]),Path(str(rec["output"])))
        try:
            scene=(lum,ts,masks,intrinsics)
            with tempfile.TemporaryDirectory(prefix="ami_baselines_") as td:
                external={m:_external(m,lum,ts,a,Path(td)) for m in ("dvs_voltmeter","v2e")}
            streams={m:[_slice(e,float(ts[i]),float(ts[i+1])) for i in range(len(ts)-1)] for m,e in external.items()}; streams["ours"]=_ours(scene,oracle,objective,model,ck,a.margin,a.confidence,a.dwell)
            for method,windows in streams.items():
                previous=None
                for i,events in enumerate(windows):
                    metrics,previous=_window_metrics(events,lum[i],masks[i],previous); rows.append({"scene":rec["scene"],"condition":rec["condition"],"window":i,"method":method,**metrics})
        finally: oracle.close()
        print(f"evaluated {rec['scene']}")
        completed+=1
    report={"protocol":"ami_downstream_dvs_v2e_ours_v1","split":a.split,"methods":["dvs_voltmeter","v2e","ours"],"aggregate":_aggregate(rows),"rows":rows,"fairness":"same source RGB/timestamps/window/resolution/downstream scorer; native event outputs","limitations":["Native event budgets are not yet matched.","Synthetic controlled-motion pilot; no prism hardware validation."]}; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2,sort_keys=True)); print(json.dumps(report["aggregate"],indent=2,sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
