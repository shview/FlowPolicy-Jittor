# FPJ-033 formal A10 evidence

`reach-wall-v3` used the full 512/1024/2048 channel configuration, 30 epochs, batch 64, seeds 0/1/2 and 20 rollout episodes per seed.

- success rate: **31.67% ± 16.07%** (20%, 25%, 50%)
- mean policy latency: **3.775 ms**, NFE=1
- hardware: NVIDIA A10, 23028 MiB, driver 570.211.01
- sampled peak GPU memory: 10605 MiB
- all 60 episodes ended at the declared 200-step cap; success is an episode-level observation and does not end the rollout early

| seed | success | last train loss | val loss | latency ms | reload max abs delta |
| --- | --- | --- | --- | --- | --- |
| 0 | 4/20 (20%) | 0.000112451 | 0.000105068 | 3.816 | 2.38e-07 |
| 1 | 5/20 (25%) | 0.00011165 | 0.000106791 | 3.756 | 1.79e-07 |
| 2 | 10/20 (50%) | 4.67114e-06 | 5.19853e-06 | 3.754 | 3.39e-05 |

## Interpretation boundary

This is a full-channel, limited-budget Jittor run in the documented modern MetaWorld/MuJoCo substitute stack. It is not the paper's 3000-epoch, 37-task recipe, does not use the paper-style detached EMA target, and must not be compared numerically with the paper success-rate table.
