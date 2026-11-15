# Third-party notices

## FlowPolicy

Parts of the module structure, tensor contracts and model hyperparameters are
derived from the official FlowPolicy implementation:

- Repository: <https://github.com/zql-kk/FlowPolicy>
- Audited commit: `86d10c14972b4eab083abec9026749c340f8e5ed`
- Original copyright: Copyright (c) 2025 zql-kk
- License: MIT; the full text is included in `LICENSE`.

Files currently derived from or designed to remain compatible with FlowPolicy:

- `src/flowpolicy_jittor/model/conv1d.py`
- `tests/test_conv1d.py`
- `tests/test_conv1d_parity.py`

## Jittor

The local `ConvTranspose1d` compatibility wrapper follows the singleton-width
strategy used by Jittor's official `Conv1d` implementation and calls Jittor's
official two-dimensional `nn.ConvTranspose` operator:

- Repository: <https://github.com/Jittor/jittor>
- API-audit commit: `ab31f57b351fccbf3a359698185d5cda8f49abba`
- Relevant source: `python/jittor/nn.py`

Jittor remains an external dependency; its source is not vendored here.

