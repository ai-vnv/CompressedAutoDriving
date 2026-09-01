# F12 Compression Ablation

All A0--A7 comparisons below use the frozen compression-selection split. `PASS`
requires both action-fidelity and C4 closed-loop gates. Final holdout was not used
to rank these variants.

| ID | Model | Precision | Params | Bytes | Median µs | v MAE | ω MAE | Completion | Selection class |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| A0 | B-PPO | FP32 | 73,986 | 299,667 | 42.77 | 0.00000 | 0.00000 | 100% | REFERENCE |
| A1 | B-PPO-P | FP32 | 6,210 | 28,883 | 34.45 | 0.03504 | 0.35821 | 0% | FAILED |
| A2 | B-PPO-PD | FP32 | 6,210 | 29,077 | 34.59 | 0.00122 | 0.01908 | 100% | PASS |
| A3 | B-PPO-Q | INT8 | 73,986 | 109,160 | 16.63 | 0.00337 | 0.07031 | 100% | FAILED |
| A4 | B-PPO-QD | INT8 | 73,986 | 111,928 | 16.44 | 0.00213 | 0.03968 | 100% | FAILED |
| A5 | B-PPO-PQ | INT8 | 6,210 | 36,856 | 13.92 | 0.03530 | 0.35715 | 0% | FAILED |
| A6 | B-PPO-PDQ | INT8 | 6,210 | 36,856 | 14.29 | 0.00242 | 0.03933 | 100% | PASS |
| A7 | B-PPO-PDQD | INT8 | 6,210 | 36,880 | 14.07 | 0.00215 | 0.03535 | 100% | PASS |

Full machine-readable tables: `artifacts/f12_belief_ppo_compression_v1/final/ablation_table.csv` and `pruning_level_table.csv`.
