# F9d execution ledger

F9d config became read-only at Task 5 with SHA256
`7bbe6525c24e294b55a46808301249633236658814e906a68d0d804d5e8a8ca6`.
F9c remained byte-identical at
`359dc52020421c248bf6c26e036234191b2b97d24b505a66d85daa85b563704e`.

| Task | Status | Durable evidence |
|---|---|---|
| 1 protocol/hash guards | COMPLETE | `f9d_protocol.py`, protocol tests |
| 2 cache-only association | COMPLETE | `f9d_association_diagnostic.json`; C1 selection REFUTED, abstention SUPPORTED; C2 NOT INFERIOR |
| 3 natural-outlier yield | COMPLETE | dev projection 119.5 frames; freeze authorised |
| 4 B1/B2/B3 absence feasibility | COMPLETE | all three dev support gates passed; B2 contamination 0; real B3 RGB/truth removal tests passed |
| 5 freeze/verifier | COMPLETE | frozen artifact, config hash, strict-failure unit test |
| 6 final outlier run | COMPLETE ONCE | 7,658 rows; 43 frames/29 events/4 seeds; `INSUFFICIENT_EVIDENCE` |
| 7 final absence run | COMPLETE ONCE | 7,260 rows; B1/B2/B3 support passed; B1/B2 criteria passed |
| 8 leakage/report/classification | COMPLETE | strict verifier 15/15; F9d classification `LIMITED` |

Final active regression suite: 351 passed, 0 failed, 0 skipped (264
dependency/runtime warnings).

Once-only seed rule: 8201–8204 and 8301–8304 must not be rendered again.
The B1 metrics-only correction was recomputed from the already-written CSV;
it did not invoke the simulator, detector, or estimator.
