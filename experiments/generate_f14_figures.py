"""Generate publication-ready F14 figures from immutable tabular artifacts."""

from __future__ import annotations

import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from duckie_pomdp.explain.compression_diagnostics import ACTION_NAMES, load_f14_config, resolve_config_path
from duckie_pomdp.explain.group_shapley import GROUP_ORDER

PHASES=("nominal","lane_curve","pedestrian_relevant","stop_required","stop_satisfied")
COLORS=("#0072B2","#E69F00","#009E73","#D55E00","#CC79A7","#56B4E9")


def save(fig, root: Path, name: str) -> None:
    fig.savefig(root/f"{name}.pdf",bbox_inches="tight")
    fig.savefig(root/f"{name}.png",dpi=320,bbox_inches="tight")
    plt.close(fig)


def matrix(rows, action):
    lookup={(r["phase"],r["action"],r["group"]):r["absolute_share"] for r in rows}
    return np.asarray([[lookup[(p,action,g)] for g in GROUP_ORDER] for p in PHASES])


def heat(ax, data, title, *, signed=False):
    lim=max(float(np.max(np.abs(data))),1e-9) if signed else 1.0
    im=ax.imshow(data,aspect="auto",cmap="RdBu_r" if signed else "Blues",vmin=-lim if signed else 0,vmax=lim)
    ax.set_xticks(range(6),GROUP_ORDER,rotation=35,ha="right"); ax.set_yticks(range(5),[p.replace("_"," ") for p in PHASES])
    ax.set_title(title,loc="left",fontweight="bold")
    for i in range(5):
        for j in range(6): ax.text(j,i,f"{data[i,j]:.2f}",ha="center",va="center",fontsize=7,color="white" if abs(data[i,j])>.48*lim else "black")
    return im


def pathway(figroot, name, variants, title, axes, fidelity, behavior):
    labels=[]; data=[]
    for v in variants:
        sem=axes[v]["semantic_attribution"]; cf=axes[v]["counterfactual_functional_sensitivity"]
        labels.append(v); data.append((sem["preserved_phase_action_cells"]/10,cf["preserved_primary_cells"]/3,
            1.0 if fidelity[v].get("pass",False) else 0.0,behavior[v]["summary"]["completion_rate"]))
    fig,ax=plt.subplots(figsize=(7.0,3.7)); x=np.arange(len(labels)); width=.18
    names=("Semantic cells","Functional cells","Fidelity gate","C4 completion")
    for k in range(4): ax.bar(x+(k-1.5)*width,[r[k] for r in data],width,label=names[k],color=COLORS[k],edgecolor="black",linewidth=.4,hatch=("","//","..","xx")[k])
    ax.set_xticks(x,labels); ax.set_ylim(0,1.08); ax.set_ylabel("Preservation / pass fraction"); ax.set_title(title,loc="left",fontweight="bold"); ax.grid(axis="y",alpha=.25); ax.legend(ncol=2,frameon=False,fontsize=8)
    save(fig,figroot,name)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/f14_explainability_aware_compression_v1.toml"); a=ap.parse_args()
    cfg=load_f14_config(a.config); root=resolve_config_path(cfg,cfg["outputs"]["directory"]); figroot=root/"figures"; figroot.mkdir(parents=True,exist_ok=True)
    for old in figroot.glob("*.png"):
        raise RuntimeError(f"immutable figure already exists: {old}")
    dev=json.loads((root/"ablation_comparison_metrics.json").read_text()); final=json.loads((root/"final_comparison_metrics.json").read_text())
    axes=dev["classification_axes"]; hist=dev["historical"]; fidelity=hist["selection_fidelity"]["results"]; behavior=hist["selection_behavior"]["results"]
    summaries={v:[] for v in (f"A{i}" for i in range(8))}
    import csv
    with (root/"ablation_group_summary.csv").open() as f:
        for r in csv.DictReader(f):
            r["absolute_share"]=float(r["absolute_share"]); summaries[r["variant"]].append(r)
    fig,axs=plt.subplots(1,2,figsize=(11,4.1),constrained_layout=True)
    for ax,action in zip(axs,ACTION_NAMES): heat(ax,matrix(summaries["A0"],action),f"A0 — {action}")
    fig.colorbar(axs[0].images[0],ax=axs,shrink=.78,label="Mean absolute Group Shapley share"); save(fig,figroot,"a0_semantic_group_shapley_heatmaps")
    pathway(figroot,"compression_pathway_a0_a1_a2",("A0","A1","A2"),"Pruning and distillation recovery: A0 → A1 → A2",axes,fidelity,behavior)
    pathway(figroot,"quantization_pathway_a0_a3_a4",("A0","A3","A4"),"Quantization and QAT/KD: A0 → A3 → A4",axes,fidelity,behavior)
    pathway(figroot,"successful_deployment_pathway_a2_a6_a7",("A2","A6","A7"),"Deployable pathway: A2 → A6 → A7",axes,fidelity,behavior)
    pathway(figroot,"a1_vs_a5_pruning_dominated_failure",("A0","A1","A5"),"Pruning-only versus pruning + PTQ failure",axes,fidelity,behavior)

    fig,axs=plt.subplots(2,2,figsize=(11,8.2),constrained_layout=True)
    for row,v in enumerate(("A0","A7")):
        for col,action in enumerate(ACTION_NAMES): heat(axs[row,col],matrix(final["group_summaries"][v],action),f"{v} — {action}")
    fig.colorbar(axs[0,0].images[0],ax=axs,shrink=.75,label="Mean absolute Group Shapley share"); save(fig,figroot,"final_a0_vs_a7_phase_shapley")

    primary=(("pedestrian_absent","pedestrian_relevant","Pedestrian absent"),("stop_absent","stop_required","Stop absent"),("lane_centered","lane_curve","Lane centered"))
    cf=final["counterfactual_comparison"]
    fig,axs=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    for ai,action in enumerate(ACTION_NAMES):
        a0=[]; a7=[]
        for op,phase,_ in primary:
            r=next(x for x in cf if x["intervention"]==op and x["phase"]==phase and x["action"]==action); a0.append(r["reference_mean"]); a7.append(r["candidate_mean"])
        x=np.arange(3); axs[ai].bar(x-.18,a0,.36,label="A0",color=COLORS[0],edgecolor="black",hatch="//"); axs[ai].bar(x+.18,a7,.36,label="A7",color=COLORS[1],edgecolor="black",hatch="..")
        axs[ai].axhline(0,color="black",lw=.7); axs[ai].set_xticks(x,[p[2] for p in primary],rotation=18,ha="right"); axs[ai].set_title(action,loc="left",fontweight="bold"); axs[ai].set_ylabel("Mean counterfactual action change"); axs[ai].grid(axis="y",alpha=.25)
    axs[0].legend(frameon=False); save(fig,figroot,"final_a0_vs_a7_counterfactual")

    level=np.zeros((8,5),dtype=float)
    H=json.loads((root/"failure_modes/failure_hierarchy.json").read_text())["classification"]
    for i,v in enumerate(H):
        labels=H[v]["observed_levels"]
        for j in range(1,5): level[i,j]=float(any(s.startswith(f"L{j}") for s in labels))
        level[i,0]=0
    fig,ax=plt.subplots(figsize=(8,4)); im=ax.imshow(level,aspect="auto",cmap="Greys",vmin=0,vmax=1)
    ax.set_xticks(range(5),("L0 integrity","L1 attribution","L2 sensitivity","L3 fidelity","L4 closed loop"),rotation=25,ha="right"); ax.set_yticks(range(8),[f"A{i}" for i in range(8)])
    ax.set_title("Descriptive failure hierarchy (presence, not causality)",loc="left",fontweight="bold")
    for i in range(8):
        for j in range(5): ax.text(j,i,"PASS" if j==0 else ("●" if level[i,j] else "–"),ha="center",va="center",color="white" if level[i,j] else "black",fontsize=8)
    save(fig,figroot,"failure_hierarchy_summary")
    print(json.dumps({"classification":"PASS","figure_pairs":8,"output":str(figroot)},indent=2))

if __name__=="__main__": main()
