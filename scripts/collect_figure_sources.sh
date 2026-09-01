#!/usr/bin/env bash
cd "$HOME/aivnv/duckie-pomdp" || exit 1
SRC17=artifacts/f17_optimization_method_order_v1
SRC18=artifacts/f18_fp16_control_v1
DST=paper/figure_sources
mkdir -p "$DST"

cp() { command cp -f "$1" "$DST/$2" && echo "  + $2"; }

echo "=== Fig T / Fig 0 insets — environment frames per curriculum ==="
cp "$SRC17/primary_media/A1/c0/seed_180201/primary_rollout_contact_sheet.png" figT_c0_A1_seed180201_contactsheet.png
cp "$SRC17/primary_media/A0/c1/seed_180206/primary_rollout_contact_sheet.png" figT_c1_A0_seed180206_contactsheet.png
cp "$SRC17/primary_media/A0/c2/seed_180207/primary_rollout_contact_sheet.png" figT_c2_A0_seed180207_contactsheet.png
cp "$SRC17/primary_media/A6/c3/seed_180201/primary_rollout_contact_sheet.png" figT_c3_A6_seed180201_contactsheet.png
cp "$SRC17/primary_media/A1/c4/seed_180201/primary_rollout_contact_sheet.png" figT_c4_A1_seed180201_contactsheet.png

echo "=== Fig 2 — A6 freeze footage (film strip source) ==="
cp "$SRC17/primary_media/A6/c3/seed_180201/primary_rollout.mp4"               fig2_A6_c3_seed180201_freeze.mp4
cp "$SRC17/primary_media/A6/c3/seed_180201/primary_rollout.gif"               fig2_A6_c3_seed180201_freeze.gif
cp "$SRC17/primary_media/A6/c4/seed_180203/primary_rollout_contact_sheet.png" fig2_A6_c4_seed180203_contactsheet.png

echo "=== referensi data-figure (bukan untuk dipakai langsung; acuan saat menggambar ulang) ==="
cp "$SRC17/figures/06_optimization_method_pathways.png"        reference_fig1_pathway_matrix.png
cp "$SRC17/figures/11_failure_state_action_table.png"          reference_fig2_failure_states.png
cp "$SRC18/figures/12_precision_control_fp32_fp16_int8.png"    reference_fig3_precision_control.png
cp "$SRC17/figures/07_distillation_coverage_recovery.png"      reference_extra_kd_coverage.png

echo; echo "=== isi folder ==="
ls -la "$DST" | awk 'NR>1{printf "  %8d  %s\n",$5,$9}'
du -sh "$DST" | awk '{print "  total: "$1}'
