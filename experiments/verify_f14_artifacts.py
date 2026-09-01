"""Independent fail-closed verifier for the completed immutable F14 package."""

from __future__ import annotations

import argparse, csv, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from duckie_pomdp.explain.compression_diagnostics import counterfactual_preservation_classification, file_sha256, load_f14_config, load_frozen_actors, load_policy_contract, resolve_config_path


def rows(path: Path) -> int:
    with path.open(newline="",encoding="utf-8") as f: return sum(1 for _ in csv.reader(f))-1


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/f14_explainability_aware_compression_v1.toml"); ap.add_argument("--write-manifest",action="store_true"); a=ap.parse_args()
    cfg=load_f14_config(a.config); root=resolve_config_path(cfg,cfg["outputs"]["directory"]); repo=Path(__file__).resolve().parents[1]
    _,names,groups=load_policy_contract(cfg); actors=load_frozen_actors(cfg)
    checks={"actor_set":set(actors)=={f"A{i}" for i in range(8)},"observation_29":len(names)==29,
        "partition_exact":sorted(i for indexes in groups.values() for i in indexes)==list(range(29))}
    historical=json.loads((root/"integrity/historical_integrity_manifest.json").read_text())
    checks["historical_immutable"]=all(file_sha256(repo/p)==h for p,h in historical["files"].items())
    cal=json.loads((root/"reference_calibration_metrics.json").read_text()); checks["reference_calibration_pass"]=cal["classification"]=="PASS" and not cal["a1_a7_results_inspected"]
    checks["local_accuracy_calibration"]=cal["maximum_local_accuracy_residual"]<=float(cfg["shapley"]["efficiency_absolute_tolerance"])
    counts={"ablation_shapley.csv":48000,"ablation_counterfactuals.csv":48000,"final_a0_a7_shapley.csv":105600,"final_a0_a7_counterfactuals.csv":105600}
    for name,count in counts.items(): checks[f"rows_{name}"]=rows(root/name)==count
    dev=np.load(root/"diagnostic/ablation_counterfactuals.npz",allow_pickle=False); final=np.load(root/"final_a0_a7_attribution.npz",allow_pickle=False)
    sham=list(dev["interventions"].astype(str)).index("sham"); checks["sham_exact_zero"]=float(np.max(np.abs(dev["effects"][:,sham])))==0.0
    checks["final_shape"]=final["attribution"].shape==(2,4400,2,6)
    final_metrics=json.loads((root/"final_comparison_metrics.json").read_text()); checks["final_no_rerun"]=final_metrics["rerendered_or_recollected"] is False
    checks["final_sources_exact"]=final_metrics["source_trace_sha256"]==cfg["final"]["factual_trace_sha256"] and final_metrics["reference_assignment_sha256"]==cfg["final"]["reference_draws_sha256"]
    figures=("a0_semantic_group_shapley_heatmaps","compression_pathway_a0_a1_a2","quantization_pathway_a0_a3_a4","successful_deployment_pathway_a2_a6_a7","a1_vs_a5_pruning_dominated_failure","final_a0_vs_a7_phase_shapley","final_a0_vs_a7_counterfactual","failure_hierarchy_summary")
    checks["figure_pairs"]=all((root/"figures"/f"{n}.png").stat().st_size>10000 and (root/"figures"/f"{n}.pdf").stat().st_size>1000 for n in figures)
    required_docs=("F14_PROTOCOL.md","F14_REFERENCE_CALIBRATION.md","F14_ABLATION_EXPLANATION.md","F14_FAILURE_MODE_REPORT.md","F14_FINAL_REEXPLANATION.md","F14_FINAL_REPORT.md")
    checks["reports_complete"]=all((repo/"docs"/n).exists() for n in required_docs)
    if not all(checks.values()): raise RuntimeError(f"F14 verification failed: {[k for k,v in checks.items() if not v]}")
    semantic=final_metrics["semantic_attribution"]["classification"]
    cf=final_metrics["counterfactual_comparison"]
    cf_status=counterfactual_preservation_classification(cf,cfg["counterfactual"])
    classification={"schema_version":1,"config_sha256":cfg["_sha256"],"overall":"LIMITED","integrity":"PASS","efficiency":"PASS","c4_behavior":"PRESERVED","semantic_attribution":semantic,"counterfactual_functional_sensitivity":cf_status["classification"],"counterfactual_primary_cells_preserved":cf_status["preserved_primary_cells"],"counterfactual_primary_cells_total":cf_status["total_primary_cells"],"retention":"NOT PRESERVED","semantic_retention":"UNRESOLVED"}
    classification_path=root/"final/f14_classification.json"
    if not classification_path.exists():
        classification_path.parent.mkdir(parents=True,exist_ok=True); classification_path.write_text(json.dumps(classification,indent=2,sort_keys=True)+"\n")
    elif json.loads(classification_path.read_text())!=classification: raise RuntimeError("frozen classification mismatch")
    if a.write_manifest:
        manifest_path=root/"artifact_manifest.json"
        if manifest_path.exists(): raise RuntimeError("artifact manifest already exists")
        files={}
        for base in (root,repo/"docs"):
            for p in sorted(base.rglob("*")):
                if p.is_file() and p!=manifest_path and "logs" not in p.parts and (base==root or p.name.startswith("F14_")):
                    files[str(p.relative_to(repo))]={"sha256":file_sha256(p),"bytes":p.stat().st_size}
        source_paths = (
            repo/"configs/f14_explainability_aware_compression_v1.toml",
            repo/"src/duckie_pomdp/explain/group_shapley.py",
            repo/"src/duckie_pomdp/explain/compression_diagnostics.py",
            repo/"experiments/prepare_f14_diagnostic.py",
            repo/"experiments/calibrate_f14_shapley_references.py",
            repo/"experiments/run_f14_ablation_explanations.py",
            repo/"experiments/analyze_f14_failure_modes.py",
            repo/"experiments/run_f14_final_reexplanation.py",
            repo/"experiments/generate_f14_figures.py",
            repo/"experiments/verify_f14_artifacts.py",
            repo/"tests/test_f14_explainability_aware_compression.py",
        )
        for p in source_paths:
            files[str(p.relative_to(repo))]={"sha256":file_sha256(p),"bytes":p.stat().st_size}
        manifest_path.write_text(json.dumps({"schema_version":1,"created_at_utc":datetime.now(timezone.utc).isoformat(),"config_sha256":cfg["_sha256"],"files":files},indent=2,sort_keys=True)+"\n")
    manifest_path=root/"artifact_manifest.json"
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        stale=[name for name,entry in manifest["files"].items() if not (repo/name).exists() or file_sha256(repo/name)!=entry["sha256"]]
        if stale: raise RuntimeError(f"artifact manifest hash mismatch: {stale[:5]}")
    print(json.dumps({"classification":"PASS","checks":checks,"f14":classification,"artifact_manifest":(root/"artifact_manifest.json").exists()},indent=2))

if __name__=="__main__": main()
