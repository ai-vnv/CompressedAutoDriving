"""Final shared model-agnostic A0 versus A7 re-explanation on frozen R004 rows."""

from __future__ import annotations

import argparse, csv, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from duckie_pomdp.explain.compression_diagnostics import (
    ACTION_NAMES, compare_group_summaries, counterfactual_comparison,
    evaluate_semantic_counterfactuals, file_sha256, load_f14_config,
    load_frozen_actors, load_policy_contract, resolve_config_path,
    semantic_structure_classification, summarize_group_attribution,
)
from duckie_pomdp.explain.group_shapley import GROUP_ORDER, exact_group_shapley


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/f14_explainability_aware_compression_v1.toml"); args=ap.parse_args()
    cfg=load_f14_config(args.config); root=resolve_config_path(cfg,cfg["outputs"]["directory"])
    out=root/"final_comparison_metrics.json"
    if out.exists(): raise RuntimeError(f"immutable final output exists: {out}")
    dev=json.loads((root/"ablation_comparison_metrics.json").read_text())
    frozen=json.loads((root/"calibration/frozen_thresholds.json").read_text())
    if frozen["reference_calibration_classification"]!="PASS" or set(dev["classification_axes"])!={f"A{i}" for i in range(8)}:
        raise RuntimeError("development diagnosis is not frozen and complete")
    protocol,names,groups=load_policy_contract(cfg); actors=load_frozen_actors(cfg)
    section=cfg["final"]
    trace_path=resolve_config_path(cfg,section["factual_trace"]); refs_path=resolve_config_path(cfg,section["reference_draws"])
    if file_sha256(trace_path)!=section["factual_trace_sha256"] or file_sha256(refs_path)!=section["reference_draws_sha256"]:
        raise RuntimeError("R004 public source provenance mismatch")
    trace=np.load(trace_path,allow_pickle=False); refs=np.load(refs_path,allow_pickle=False)
    index=refs["sample_index"].astype(np.int64); x=refs["observation"].astype(np.float32)
    physical=trace["physical_observation"][index].astype(np.float32); phases=refs["public_phase"].astype(str)
    references=refs["reference_observation"].astype(np.float32)
    if references.shape!=(6,4,4400,29) or not np.array_equal(trace["seed"][index],refs["seed"]):
        raise RuntimeError("R004 factual/reference alignment mismatch")
    try: import tomllib
    except ModuleNotFoundError: import tomli as tomllib
    cf_path=resolve_config_path(cfg,cfg["counterfactual"]["source_config"])
    if file_sha256(cf_path)!=cfg["counterfactual"]["source_config_sha256"]: raise RuntimeError("R003 operator provenance mismatch")
    with cf_path.open("rb") as f: icfg=tomllib.load(f)["r003"]
    interventions=tuple(cfg["counterfactual"]["interventions"])
    means={}; summaries={}; effects={}; actions={}; residual={}
    for variant in ("A0","A7"):
        result=exact_group_shapley(actors[variant].physical,x,references,groups,names,
            observation_clip=float(cfg["frozen"]["observation_clip"]), state_batch_size=int(cfg["shapley"]["state_batch_size"]))
        residual[variant]=float(np.max(np.abs(result.efficiency_residual)))
        if residual[variant]>float(cfg["shapley"]["efficiency_absolute_tolerance"]): raise RuntimeError(f"{variant} local accuracy failure")
        means[variant]=result.mean_attribution; summaries[variant]=summarize_group_attribution(result.mean_attribution,phases)
        factual,effect,_=evaluate_semantic_counterfactuals(actors[variant],x,physical,protocol,interventions,
            lane_low_confidence_validity=float(icfg["lane_low_confidence_validity"]),
            lane_low_confidence_min_lateral_std_m=float(icfg["lane_low_confidence_min_lateral_std_m"]),
            lane_low_confidence_min_heading_std_rad=float(icfg["lane_low_confidence_min_heading_std_rad"]),
            lane_low_confidence_min_curvature_std_inv_m=float(icfg["lane_low_confidence_min_curvature_std_inv_m"]))
        if np.max(np.abs(effect[interventions.index("sham")]))>float(cfg["counterfactual"]["sham_absolute_tolerance"]): raise RuntimeError(f"{variant} sham failed")
        effects[variant]=effect; actions[variant]=factual
    np.savez_compressed(root/"final_a0_a7_attribution.npz", variants=np.asarray(("A0","A7")),
        attribution=np.stack((means["A0"],means["A7"])), factual_action=np.stack((actions["A0"],actions["A7"])),
        phase=phases,seed=refs["seed"],step=refs["step"],group_names=np.asarray(GROUP_ORDER),action_names=np.asarray(ACTION_NAMES))
    rows=[]
    for variant in ("A0","A7"):
        for i in range(len(x)):
            absolute=np.abs(means[variant][i]); share=absolute/np.maximum(absolute.sum(axis=1,keepdims=True),1e-12)
            for ai,a in enumerate(ACTION_NAMES):
                for gi,g in enumerate(GROUP_ORDER): rows.append({"variant":variant,"actor_sha256":actors[variant].sha256,"state_id":i,"seed":int(refs["seed"][i]),"step":int(refs["step"][i]),"phase":phases[i],"action":a,"group":g,"signed_shapley":float(means[variant][i,ai,gi]),"absolute_shapley":float(absolute[ai,gi]),"absolute_share":float(share[ai,gi])})
    write_csv(root/"final_a0_a7_shapley.csv",rows)
    cfrows=[]
    for variant in ("A0","A7"):
        for ii,inter in enumerate(interventions):
            for i in range(len(x)):
                for ai,a in enumerate(ACTION_NAMES): cfrows.append({"variant":variant,"actor_sha256":actors[variant].sha256,"state_id":i,"seed":int(refs["seed"][i]),"step":int(refs["step"][i]),"phase":phases[i],"intervention":inter,"action":a,"factual_action":float(actions[variant][i,ai]),"counterfactual_action":float(actions[variant][i,ai]+effects[variant][ii,i,ai]),"effect":float(effects[variant][ii,i,ai])})
    write_csv(root/"final_a0_a7_counterfactuals.csv",cfrows)
    comparison=compare_group_summaries(summaries["A0"],summaries["A7"],signed_deadband=float(cfg["shapley"]["signed_deadband"]))
    semantic=semantic_structure_classification(comparison,frozen["thresholds"])
    cf=counterfactual_comparison(effects["A0"],effects["A7"],phases,interventions,direction_deadband=float(cfg["counterfactual"]["direction_deadband"]))
    payload={"schema_version":1,"created_at_utc":datetime.now(timezone.utc).isoformat(),"config_sha256":cfg["_sha256"],
        "source_trace_sha256":file_sha256(trace_path),"reference_assignment_sha256":file_sha256(refs_path),
        "actors":{v:actors[v].sha256 for v in ("A0","A7")},"states":len(x),"draws":6,"references_per_draw":4,
        "maximum_local_accuracy_residual":residual,"semantic_attribution":semantic,"counterfactual_comparison":cf,
        "group_summaries":{"A0":summaries["A0"],"A7":summaries["A7"]},
        "historical_f12_c4_behavior":"PRESERVED","historical_f12_retention":"LIMITED","rerendered_or_recollected":False}
    write_json(out,payload)
    print(json.dumps({"classification":"PASS","semantic":semantic["classification"],"preserved_cells":semantic["preserved_phase_action_cells"],"max_residual":residual},indent=2))

if __name__=="__main__": main()
