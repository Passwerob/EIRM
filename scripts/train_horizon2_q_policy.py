#!/usr/bin/env python3
"""Train a causal GRU to distill privileged horizon-2 action returns."""
from __future__ import annotations
import argparse, copy, json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from ami_simulator.motion_net import CausalAnchorQGRUPolicy

def _windows(x, y, length):
    if len(x) < length:
        return np.empty((0, length, x.shape[-1]), np.float32), np.empty((0, y.shape[-1]), np.float32)
    return np.stack([x[i-length+1:i+1] for i in range(length-1, len(x))]).astype(np.float32), np.stack([y[i] for i in range(length-1, len(y))]).astype(np.float32)

def _load(manifest, split, length):
    xs=[]; ys=[]; acts=[]; reports=[]
    for record in manifest["records"]:
        if record["split"] != split: continue
        with np.load(Path(str(record["output"]))) as data:
            x,y = _windows(np.asarray(data["features"],np.float32), np.asarray(data["q_h2"],np.float32), length)
            a=np.asarray(data["teacher_action"],np.int64)[length-1:]
        if len(x)!=len(a): raise ValueError(f"label length mismatch: {record['output']}")
        xs.append(x); ys.append(y); acts.append(a); reports.append({"scene":record["scene"],"condition":record["condition"],"windows":int(len(y))})
    if not xs: raise ValueError(f"No horizon-2 records for split {split!r}")
    return np.concatenate(xs),np.concatenate(ys),np.concatenate(acts),reports

def _norm(train, values):
    mean=train.mean(axis=(0,1),keepdims=True).astype(np.float32); std=train.std(axis=(0,1),keepdims=True).astype(np.float32); std=np.where(std<1e-4,1.0,std)
    return np.clip((train-mean)/std,-5,5),np.clip((values-mean)/std,-5,5),mean,std

def _metrics(q,target,anchor):
    action=np.argmax(q,axis=1); selected=target[np.arange(len(target)),action]; oracle=target.max(axis=1)
    return {"mean_selected_return":float(selected.mean()),"mean_oracle_return":float(oracle.mean()),"mean_regret":float((oracle-selected).mean()),"anchor_return":float(anchor.mean()),"gain_vs_anchor":float((selected-anchor).mean()),"anchor_action_fraction":float(np.mean(action==1)),"non_anchor_fraction":float(np.mean(action!=1))}

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--window",type=int,default=8); p.add_argument("--hidden-dim",type=int,default=64); p.add_argument("--epochs",type=int,default=400); p.add_argument("--lr",type=float,default=2e-3); p.add_argument("--classification-weight",type=float,default=.25); p.add_argument("--seed",type=int,default=17); a=p.parse_args()
    if a.window<2 or a.hidden_dim<4 or a.epochs<1 or a.lr<=0 or a.classification_weight<0: p.error("invalid training arguments")
    torch.manual_seed(a.seed); np.random.seed(a.seed); manifest=json.loads(a.manifest.read_text(encoding="utf-8"))
    train_x,train_y,train_a,train_reports=_load(manifest,"train",a.window); valid_x,valid_y,valid_a,valid_reports=_load(manifest,"valid",a.window); train_x,valid_x,mean,std=_norm(train_x,valid_x)
    model=CausalAnchorQGRUPolicy(feature_dim=train_x.shape[-1],hidden_dim=a.hidden_dim,num_actions=train_y.shape[-1]); opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4); tx,ty,ta=torch.from_numpy(train_x),torch.from_numpy(train_y),torch.from_numpy(train_a); vx,vy,va=torch.from_numpy(valid_x),torch.from_numpy(valid_y),torch.from_numpy(valid_a)
    best=copy.deepcopy(model.state_dict()); best_loss=float("inf"); best_epoch=0; history=[]
    for epoch in range(a.epochs+1):
        model.train(); opt.zero_grad(set_to_none=True); pred=model(tx); loss_q=nn.functional.smooth_l1_loss(pred,ty); loss_cls=nn.functional.cross_entropy(pred,ta); loss=loss_q+a.classification_weight*loss_cls
        if epoch>0: loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
        model.eval();
        with torch.inference_mode(): vp=model(vx); val=float((nn.functional.smooth_l1_loss(vp,vy)+a.classification_weight*nn.functional.cross_entropy(vp,va)).item())
        if val<best_loss: best_loss=val; best=copy.deepcopy(model.state_dict()); best_epoch=epoch
        if epoch in (0,10,25,50,100,200,a.epochs): history.append({"epoch":epoch,"train_loss":float(loss.item()),"valid_loss":val})
    model.load_state_dict(best); model.eval();
    with torch.inference_mode(): train_pred=model(tx).numpy(); valid_pred=model(vx).numpy()
    report={"architecture":"anchor_q_gru","target_key":"q_h2","feature_dim":int(train_x.shape[-1]),"hidden_dim":a.hidden_dim,"window":a.window,"candidate_rpm":[0.,180.,360.],"anchor_index":1,"best_epoch":best_epoch,"best_validation_loss":best_loss,"history":history,"train":{"samples":len(train_y),**_metrics(train_pred,train_y,train_y[:,1])},"valid":{"samples":len(valid_y),**_metrics(valid_pred,valid_y,valid_y[:,1])},"data":{"train_scenes":len(train_reports),"valid_scenes":len(valid_reports),"train_reports":train_reports,"valid_reports":valid_reports}}
    ckpt={"model":model.state_dict(),"architecture":"anchor_q_gru","feature_dim":int(train_x.shape[-1]),"hidden_dim":a.hidden_dim,"num_actions":int(train_y.shape[-1]),"candidate_rpm":[0.,180.,360.],"anchor_index":1,"window":a.window,"feature_mean":mean.reshape(-1).tolist(),"feature_std":std.reshape(-1).tolist(),"feature_clip":5.,"report":str(a.output.with_suffix('.json'))}
    a.output.parent.mkdir(parents=True,exist_ok=True); torch.save(ckpt,a.output); a.output.with_suffix('.json').write_text(json.dumps(report,indent=2,sort_keys=True),encoding="utf-8"); print(json.dumps(report,indent=2,sort_keys=True)); return 0

if __name__=="__main__": raise SystemExit(main())
