# FPJ-036 A10 paired target comparison

- Commit: `a24ae5581aa28969b4cbe2c8dfa052145b750cf0`
- Hardware: NVIDIA A10 23,028 MiB; Jittor 1.3.11.0; CUDA execution
- Task: `reach-wall-v3`; seed 0; 30 epochs; batch 64; 20 rollout episodes; 200-step cap
- Architecture: down dims 512/1024/2048, embed dim 128, PointNet out 64, NFE 1
- Only compared variable: released-code target semantics vs detached EMA target semantics

## Result

- `code`: 10/20 successful episodes (50%); mean policy latency 3.562 ms
- `detached`: 6/20 successful episodes (30%); mean policy latency 3.502 ms
- Peak sampled GPU memory: code train 11053 MiB; detached train 11741 MiB; rollout 8721 MiB
- Final validation loss: code 4.84726615e-06; detached 0.000111669076
- Checkpoint hashes were independently re-read after the run and all recorded hashes matched.

## Interpretation boundary

This is one paired seed, not a three-seed statistical result. It supports the narrower statement that both target variants train and run end-to-end on the full-channel A10 setup, and that seed 0 produced 50% versus 30% success in this task. It does not establish a general ranking or reproduce paper-scale results. Every episode ran to the 200-step cap; `success_steps` counts successful observations rather than an early termination time.
