# Matched target-semantics experiment (FPJ-035)

- target modes: code, detached
- tasks: reach-wall-v3
- seeds per task: [0, 1, 2]
- episodes per seed: 20
- training budget: 10 epochs, batch 32, down_dims [32, 64], device cpu
- rollout step cap: 200, NFE 1, action_slice_mode `code`

- matched preflight SHA-256: `a02835559c7daac866c47afb946128f2c8c0c5c4fa18280f8aaac36a5fd21eec`

| target | task | seeds ok | success mean ± sample std | val loss mean ± sample std | train ms/step |
| --- | --- | --- | --- | --- | --- |
| code | reach-wall-v3 | 3/3 | 51.67% ± 5.77% | 1.19e-05 ± 5.48e-06 | 723.31 |
| detached | reach-wall-v3 | 3/3 | 16.67% ± 15.28% | 2.38e-05 ± 1.04e-06 | 884.41 |

## Per-seed cells

| target | task | seed | status | success | val loss | train ms/step |
| --- | --- | --- | --- | --- | --- | --- |
| code | reach-wall-v3 | 0 | ok | 9/20 | 1.08e-05 | 718.65 |
| code | reach-wall-v3 | 1 | ok | 11/20 | 7.08e-06 | 733.91 |
| code | reach-wall-v3 | 2 | ok | 11/20 | 1.79e-05 | 717.36 |
| detached | reach-wall-v3 | 0 | ok | 6/20 | 2.4e-05 | 866.17 |
| detached | reach-wall-v3 | 1 | ok | 4/20 | 2.27e-05 | 895.76 |
| detached | reach-wall-v3 | 2 | ok | 0/20 | 2.48e-05 | 891.31 |

## What this does not mean

- CPU only: CUDA was not used for this grid, so no GPU throughput claim is made.
- This comparison isolates target semantics only within the declared reduced setup.
- Matching the paper architecture is not the paper's recipe; batch, epochs, data and simulator must be read from the report.
- The simulator is a documented substitute (D016); success rates cannot be compared to the paper's numbers.
- Ten to twenty demonstrations overfit quickly; a low rate is a data/scale statement, not a code verdict.
