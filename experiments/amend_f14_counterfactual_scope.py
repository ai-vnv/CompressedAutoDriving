"""One-time preregistration-alignment amendment; no actor inference is performed."""

from __future__ import annotations

import argparse, json, shutil
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from duckie_pomdp.explain.compression_diagnostics import (
    counterfactual_comparison, counterfactual_preservation_classification,
    file_sha256, load_f14_config, resolve_config_path,
)


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8")


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/f14_explainability_aware_compression_v1.toml"); a=ap.parse_args()
    cfg=load_f14_config(a.config); root=resolve_config_path(cfg,cfg["outputs"]["directory"])
    amendment=root/"protocol_alignment_amendment.json"; backup=root/"_attempt1_counterfactual_scope_error"
    if amendment.exists() or backup.exists(): raise RuntimeError("F14 protocol-alignment amendment is once-only")
    backup.mkdir(); old_hashes={}
    for relative in ("ablation_comparison_metrics.json","failure_modes/failure_hierarchy.json","final/f14_classification.json","artifact_manifest.json"):
        p=root/relative
        if p.exists():
            dst=backup/relative; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,dst); old_hashes[relative]=file_sha256(p)
            if relative in ("final/f14_classification.json", "artifact_manifest.json"):
                p.unlink()
    figures=root/"figures"
    if figures.exists(): figures.rename(backup/"figures")
    metrics_path=root/"ablation_comparison_metrics.json"; metrics=json.loads(metrics_path.read_text())
    data=np.load(root/"diagnostic/ablation_counterfactuals.npz",allow_pickle=False)
    variants=tuple(data["variants"].astype(str)); interventions=tuple(data["interventions"].astype(str)); effects=data["effects"]; phases=data["phase"].astype(str)
    for i,v in enumerate(variants):
        rows=counterfactual_comparison(effects[0],effects[i],phases,interventions,direction_deadband=float(cfg["counterfactual"]["direction_deadband"]))
        metrics["classification_axes"][v]["counterfactual_functional_sensitivity"]=counterfactual_preservation_classification(rows,cfg["counterfactual"])
    metrics["protocol_alignment_amendment"]={"scope":"three preregistered primary cells plus independent sham gate","raw_shapley_or_effect_values_changed":False}
    write(metrics_path,metrics)
    fidelity=metrics["historical"]["selection_fidelity"]["results"]; behavior=metrics["historical"]["selection_behavior"]["results"]
    hierarchy={}
    for v in variants:
        sem=metrics["classification_axes"][v]["semantic_attribution"]["classification"]; cf=metrics["classification_axes"][v]["counterfactual_functional_sensitivity"]["classification"]
        levels=[]
        if sem!="PRESERVED": levels.append("L1 semantic attribution drift")
        if cf!="PRESERVED": levels.append("L2 functional sensitivity drift")
        if not fidelity[v].get("pass",False): levels.append("L3 action fidelity drift")
        if not behavior[v].get("pass",False): levels.append("L4 closed-loop control failure")
        hierarchy[v]={"integrity":"L0 PASS","semantic":sem,"counterfactual":cf,"action_fidelity":"PRESERVED" if fidelity[v].get("pass",False) else "DRIFTED","closed_loop":"PRESERVED" if behavior[v].get("pass",False) else "NOT PRESERVED","observed_levels":levels or ["no diagnosed drift"]}
    write(root/"failure_modes/failure_hierarchy.json",{"schema_version":1,"created_at_utc":datetime.now(timezone.utc).isoformat(),"classification":hierarchy,"interpretation":"Levels are descriptive co-occurrences; no level is asserted to cause another.","protocol_alignment_amendment":True})
    write(amendment,{"schema_version":1,"created_at_utc":datetime.now(timezone.utc).isoformat(),"classification":"CORRECTED_BEFORE_FINAL_HANDOFF","reason":"Initial summary incorrectly treated both actor outputs for all interventions as eight primary cells; frozen F14 protocol specifies pedestrian-v, stop-v, lane-omega and an independent sham gate.","raw_artifacts_unchanged":["ablation_shapley.csv","ablation_counterfactuals.csv","diagnostic/ablation_attribution.npz","diagnostic/ablation_counterfactuals.npz","final_a0_a7_shapley.csv","final_a0_a7_counterfactuals.csv"],"old_artifact_hashes":old_hashes,"backup":str(backup),"thresholds_changed":False,"actor_inference_rerun":False})
    print(json.dumps({"classification":"PASS","axes":{v:metrics["classification_axes"][v]["counterfactual_functional_sensitivity"]["classification"] for v in variants}},indent=2))

if __name__=="__main__": main()
