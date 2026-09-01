"""Integrate frozen F12 evidence and explain the available pruning frontier."""

from __future__ import annotations

import argparse, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from duckie_pomdp.explain.compression_diagnostics import (
    FrozenActor, compare_group_summaries, file_sha256, load_f14_config,
    load_policy_contract, resolve_config_path, semantic_structure_classification,
    summarize_group_attribution,
)
from duckie_pomdp.explain.group_shapley import exact_group_shapley
from duckie_pomdp.optimization.actor_compression import load_dense_actor


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8")


def load_declared(config: dict, key: str) -> tuple[Path, dict]:
    p=resolve_config_path(config,config["historical"][key]); return p,json.loads(p.read_text())


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/f14_explainability_aware_compression_v1.toml"); a=ap.parse_args()
    cfg=load_f14_config(a.config); root=resolve_config_path(cfg,cfg["outputs"]["directory"])
    target=root/"failure_modes/failure_hierarchy.json"
    if target.exists(): raise RuntimeError(f"immutable analysis exists: {target}")
    protocol,names,groups=load_policy_contract(cfg); del protocol
    frozen=json.loads((root/"calibration/frozen_thresholds.json").read_text())
    ablations=json.loads((root/"ablation_comparison_metrics.json").read_text())
    states=np.load(root/"diagnostic/diagnostic_states.npz",allow_pickle=False)
    references=np.load(root/"diagnostic/reference_assignments.npz",allow_pickle=False)["observation"][:1]
    phases=states["public_phase"].astype(str); x=states["observation"].astype(np.float32)
    registry_path,registry=load_declared(cfg,"pruning_registry")
    fidelity_path,fidelity=load_declared(cfg,"pruning_fidelity")
    behavior_path,behavior=load_declared(cfg,"pruning_behavior")
    for p in (registry_path,fidelity_path,behavior_path):
        if not p.exists(): raise RuntimeError("missing frozen pruning evidence")
    variants=("A0","P192","PD192","P128","PD128","P96","PD96","P64","PD64")
    summaries={}; frontier={}
    for variant in variants:
        item=registry["candidates"][variant]
        p=Path(item["model_path"]); actual=file_sha256(p)
        if actual!=item["sha256"]: raise RuntimeError(f"frontier hash mismatch: {variant}")
        module=load_dense_actor(p)[0].cpu().eval()
        actor=FrozenActor(variant,item["name"],"FP32",p,actual,(29,*item["hidden_sizes"],2),module)
        result=exact_group_shapley(actor.physical,x,references,groups,names,observation_clip=float(cfg["frozen"]["observation_clip"]),state_batch_size=16)
        rows=summarize_group_attribution(result.mean_attribution,phases); summaries[variant]=rows
        comparison=compare_group_summaries(summaries["A0"],rows,signed_deadband=float(cfg["shapley"]["signed_deadband"]))
        semantic=semantic_structure_classification(comparison,frozen["thresholds"])
        frontier[variant]={"sha256":actual,"hidden_sizes":item["hidden_sizes"],"parameter_count":item["parameter_count"],
            "maximum_local_accuracy_residual":float(np.max(np.abs(result.efficiency_residual))),"semantic":semantic,
            "fidelity":fidelity["results"][variant],"behavior":behavior["results"][variant]}
    dump(root/"pruning_frontier_metrics.json",{"schema_version":1,"created_at_utc":datetime.now(timezone.utc).isoformat(),
        "config_sha256":cfg["_sha256"],"reference_design":"first preregistered draw, four complete same-phase references; diagnostic only",
        "states":len(x),"registry_sha256":file_sha256(registry_path),"variants":frontier})

    axes=ablations["classification_axes"]; hist=ablations["historical"]
    fidelity_all=hist["selection_fidelity"]["results"]; behavior_all=hist["selection_behavior"]["results"]
    hierarchy={}
    for variant in (f"A{i}" for i in range(8)):
        sem=axes[variant]["semantic_attribution"]["classification"]
        cf=axes[variant]["counterfactual_functional_sensitivity"]["classification"]
        fid=bool(fidelity_all[variant].get("pass",False)); beh=bool(behavior_all[variant].get("pass",False))
        levels=[]
        if sem!="PRESERVED": levels.append("L1 semantic attribution drift")
        if cf!="PRESERVED": levels.append("L2 functional sensitivity drift")
        if not fid: levels.append("L3 action fidelity drift")
        if not beh: levels.append("L4 closed-loop control failure")
        hierarchy[variant]={"integrity":"L0 PASS","semantic":sem,"counterfactual":cf,"action_fidelity":"PRESERVED" if fid else "DRIFTED",
            "closed_loop":"PRESERVED" if beh else "NOT PRESERVED","observed_levels":levels or ["no diagnosed drift"]}
    dump(target,{"schema_version":1,"created_at_utc":datetime.now(timezone.utc).isoformat(),"classification":hierarchy,
        "interpretation":"Levels are descriptive co-occurrences; no level is asserted to cause another."})
    dump(root/"failure_trace_manifest.json",{"schema_version":1,"classification":"UNRESOLVED","value":None,
        "reason":"Frozen F12 failed-ablation evidence contains episode summaries but no provenance-bound per-step public 29D trajectory/action trace for paired objective event-window selection. F14 did not rerun historical evaluations.",
        "available_summary_sources":[str(fidelity_path),str(behavior_path)],"cherry_picked":False})
    retention_path,retention=load_declared(cfg,"retention")
    dump(root/"retention_semantic_diagnostic.json",{"schema_version":1,"classification":"UNRESOLVED","value":None,
        "reason":"No compatible frozen per-step public 29D C0-C3 rows were present; historical retention evaluations were not reopened or rerendered.",
        "historical_behavior_sha256":file_sha256(retention_path),"historical_retention":retention})
    print(json.dumps({"classification":"PASS","frontier_variants":len(frontier),"failure_trace":"UNRESOLVED","retention_semantic":"UNRESOLVED"},indent=2))

if __name__=="__main__": main()
