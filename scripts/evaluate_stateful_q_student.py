#!/usr/bin/env python3
"""Evaluate a horizon-2 Q student with persistent event-sensor state."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from ami_simulator.flow import estimate_dense_flow, summarize_motion_clusters
from ami_simulator.geometry import RotatingWedgePrism
from ami_simulator.motion_net import CausalAnchorQGRUPolicy
from scripts.evaluate_anchor_residual_policy import CLUSTER_FEATURE_DIM, _state
from scripts.evaluate_phase_continuous_control import _load_scene
from scripts.run_control_headroom_stateful import RolloutState, _transition

ACTIONS = (0.0, 180.0, 360.0)

def _run(scene, oracle, objective, *, model=None, checkpoint=None, margin=.02, confidence=.50, dwell=.025):
    luminance, timestamps, masks, intrinsics = scene; prism=RotatingWedgePrism(intrinsics,2.0)
    cluster=np.stack([summarize_motion_clusters(estimate_dense_flow(luminance[i],luminance[i+1]),clusters=3) for i in range(len(timestamps)-1)]).astype(np.float32)
    state=RolloutState(0.,None,None,180.,float(timestamps[0])-dwell); states=[]; metrics=[]; rpms=[]; changes=[]
    mean=np.asarray(checkpoint["feature_mean"],np.float32) if checkpoint else None; std=np.asarray(checkpoint["feature_std"],np.float32) if checkpoint else None; clip=float(checkpoint.get("feature_clip",5.)) if checkpoint else 5.; window=int(checkpoint.get("window",8)) if checkpoint else 8
    for index in range(len(timestamps)-1):
        feature=_state(oracle,index=index,previous=state.previous_metrics,phase=state.phase,dwell_seconds=float(timestamps[index]-state.last_switch_time),previous_rpm=state.previous_rpm,cluster_features=cluster); states.append(feature); rpm=180.
        if model is not None and index>=window-1:
            history=np.clip((np.stack(states[-window:])-mean)/std,-clip,clip)
            with torch.inference_mode(): q=model(torch.from_numpy(history[None].astype(np.float32)))[0].numpy()
            best=int(np.argmax(q)); probs=torch.softmax(torch.from_numpy(q),dim=0).numpy(); advantage=float(q[best]-q[1])
            ready=float(timestamps[index]-state.last_switch_time)>=dwell
            if best!=1 and advantage>margin and float(probs[best])>confidence and ready: rpm=ACTIONS[best]
        result,next_state=_transition(scene,oracle,objective,prism,state,index,rpm,dwell=dwell); row={k:float(v) for k,v in result["metrics"].items()}; metrics.append(row); rpms.append(rpm)
        if index: changes.append(rpm!=state.previous_rpm)
        state=next_state
    return {"windows":len(metrics),"mean_edge_f1":float(np.mean([x["edge_f1"] for x in metrics])),"mean_edge_recall":float(np.mean([x["edge_coverage"] for x in metrics])),"mean_edge_precision":float(np.mean([x["edge_precision"] for x in metrics])),"mean_event_density":float(np.mean([x["event_density"] for x in metrics])),"mean_event_efficiency":float(np.mean([x["event_efficiency"] for x in metrics])),"mean_edge_event_efficiency":float(np.mean([x["edge_event_efficiency"] for x in metrics])),"mean_temporal_overlap":float(np.mean([x["temporal_overlap"] for x in metrics])),"mean_rpm":float(np.mean(rpms)),"switch_transition_fraction":float(np.mean(changes)) if changes else 0.,"anchor_deviation_fraction":float(np.mean(np.asarray(rpms)!=180.))}

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--manifest",type=Path,default=Path("results/comotion_v2_learned/manifest.json")); p.add_argument("--dataset-root",type=Path,default=Path("data/comotion_controlled_v2")); p.add_argument("--student",type=Path,required=True); p.add_argument("--split",choices=("train","valid","test"),default="test"); p.add_argument("--switch-margin",type=float,default=.02); p.add_argument("--switch-confidence",type=float,default=.50); p.add_argument("--minimum-dwell-seconds",type=float,default=.025); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    manifest=json.loads(a.manifest.read_text(encoding="utf-8")); ckpt=torch.load(a.student,map_location="cpu",weights_only=True)
    if ckpt.get("architecture")!="anchor_q_gru" or int(ckpt.get("feature_dim",0))!=CLUSTER_FEATURE_DIM: raise ValueError("Expected 47-dimensional anchor_q_gru checkpoint")
    model=CausalAnchorQGRUPolicy(feature_dim=CLUSTER_FEATURE_DIM,hidden_dim=int(ckpt["hidden_dim"]),num_actions=int(ckpt["num_actions"])); model.load_state_dict(ckpt["model"]); model.eval(); objective={**manifest["configuration"],"event_efficiency_weight":0.0}; rows=[]
    for record in manifest["oracles"]:
        if record["split"]!=a.split: continue
        luminance,timestamps,masks,intrinsics,oracle=_load_scene(a.dataset_root/str(record["scene"]),Path(str(record["output"])))
        try:
            scene=(luminance,timestamps,masks,intrinsics)
            for policy in ("fixed_180","student"):
                result=_run(scene,oracle,objective,model=model if policy=="student" else None,checkpoint=ckpt if policy=="student" else None,margin=a.switch_margin,confidence=a.switch_confidence,dwell=a.minimum_dwell_seconds); rows.append({"scene":record["scene"],"seed":int(record["seed"]),"condition":record["condition"],"policy":policy,**result})
        finally: oracle.close()
    aggregate={}
    for policy in ("fixed_180","student"):
        selected=[r for r in rows if r["policy"]==policy]; aggregate[policy]={k:float(np.mean([float(r[k]) for r in selected])) for k in ("mean_edge_f1","mean_edge_recall","mean_edge_precision","mean_event_density","mean_event_efficiency","mean_edge_event_efficiency","mean_temporal_overlap","mean_rpm","switch_transition_fraction","anchor_deviation_fraction")}
    report={"evaluation":"stateful horizon-2 Q student rollout","split":a.split,"student":str(a.student),"feature_dim":CLUSTER_FEATURE_DIM,"candidate_rpm":list(ACTIONS),"horizon":2,"switch_margin":a.switch_margin,"switch_confidence":a.switch_confidence,"minimum_dwell_seconds":a.minimum_dwell_seconds,"event_efficiency_weight":0.0,"aggregate":aggregate,"rows":rows,"limitations":["Teacher returns use privileged target edge-F1 and are used only for offline distillation.","Finite action set {0,180,360} and horizon 2; no motor ramp/readout noise model."]}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2,sort_keys=True),encoding="utf-8"); print(json.dumps(aggregate,indent=2,sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
