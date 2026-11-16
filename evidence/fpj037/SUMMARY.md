# FPJ-037 A10 three-seed target comparison

- Task: `reach-wall-v3`; seeds 0/1/2; full 512/1024/2048 channels
- Budget: 30 epochs, batch 64, 20 closed-loop episodes per seed, 200-step cap
- Compared variable: released-code target semantics vs detached EMA loss target

| target | per-seed successes | mean ± sample std | mean policy latency |
|---|---:|---:|---:|
| code | 10/20, 5/20, 2/20 | 28.33% ± 20.21% | 3.701 ms |
| detached | 6/20, 9/20, 1/20 | 26.67% ± 20.21% | 3.683 ms |

## Interpretation boundary

- This is a selected-task, 30-epoch finite-budget comparison, not the paper's 3000-epoch 37-task result.
- The modern MetaWorld/MuJoCo stack is a documented substitute for the paper's older simulator stack.
- Success rates are recomputed from every declared closed-loop episode; training loss is not used as a proxy for task success.
- The comparison isolates released-code target semantics versus a detached EMA loss target; it does not prove either semantic is universally superior.

## Evaluation-only diagnostics

| variant | target | weights | slice | seed | successes | mean latency |
|---|---|---|---|---:|---:|---:|
| code-seed1-ema-code | code | ema | code | 1 | 2/20 | 3.780 ms |
| code-seed1-online-exact | code | online | exact | 1 | 3/20 | 3.776 ms |
| detached-seed1-ema-code | detached | ema | code | 1 | 10/20 | 3.758 ms |

- These rollouts isolate weight selection or action slicing on selected seed-1 trainer checkpoints.
- They are diagnostics, not additional independent training seeds and are excluded from the three-seed target aggregate.
