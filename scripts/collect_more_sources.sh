#!/usr/bin/env bash
cd "$HOME/aivnv/duckie-pomdp" || exit 1
S17=artifacts/f17_optimization_method_order_v1/primary_media
S18=artifacts/f18_fp16_control_v1/primary_media
DST=paper/figure_sources
cpf() { command cp -f "$1" "$DST/$2" 2>/dev/null && echo "  + $2" || echo "  ! missing: $1"; }

echo "=== Fig 0 pipeline vignettes ==="
cpf "$S17/A1/c2/seed_180201/primary_rollout_contact_sheet.png" fig0_A1_c2_seed180201_lanefailure_contactsheet.png
cpf "$S17/A1/c0/seed_180201/primary_rollout_contact_sheet.png" fig0_A1_c0_seed180201_invalidpose_contactsheet.png
cpf "$S17/A2/c2/seed_180201/primary_rollout_contact_sheet.png" fig0_A2_c2_seed180201_lanefailure_contactsheet.png
cpf "$S17/A2/c0/seed_180201/primary_rollout_contact_sheet.png" fig0_A2_c0_seed180201_invalidpose_contactsheet.png
cpf "$S17/A3/c1/seed_180206/primary_rollout_contact_sheet.png" fig0_A3_c1_seed180206_benigntiles_contactsheet.png
cpf "$S17/A4/c1/seed_180206/primary_rollout_contact_sheet.png" fig0_A4_c1_seed180206_benigntiles_contactsheet.png
cpf "$S17/A4/c2/seed_180203/primary_rollout_contact_sheet.png" fig0_A4_c2_seed180203_benigntiles_contactsheet.png
cpf "$S18/F16H/c1/seed_180206/primary_rollout_contact_sheet.png" fig0_A9_c1_seed180206_benigntiles_contactsheet.png

echo "=== ekstra untuk fleksibilitas komposisi ==="
cpf "$S17/A5/c2/seed_180201/primary_rollout_contact_sheet.png" extra_A5_c2_seed180201_lanefailure_contactsheet.png
cpf "$S17/A7/c2/seed_180201/primary_rollout_contact_sheet.png" extra_A7_c2_seed180201_lanefailure_contactsheet.png
cpf "$S17/A1/c3/seed_180201/primary_rollout_contact_sheet.png" extra_A1_c3_seed180201_stopviolation_contactsheet.png
cpf "$S17/A1/c2/seed_180201/primary_rollout.mp4"               extra_A1_c2_seed180201_lanefailure.mp4
cpf "$S17/A1/c2/seed_180201/primary_rollout.gif"               extra_A1_c2_seed180201_lanefailure.gif

echo; echo "=== isi folder final ==="
ls "$DST" | sort | sed 's/^/  /'
du -sh "$DST" | awk '{print "  total: "$1}'
