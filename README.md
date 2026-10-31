# ASRINet

Core code for **ASRINet: Residual Refinement for Few-Shot Defect Segmentation**.

ASRI refines a frozen support-conditioned segmentation model using shallow query
features and prediction-derived conditions. It predicts a 16-channel spatial
residual and a shared local bias, rearranges the result with PixelShuffle, and
adds it to the foreground-background logit margin.

## Contents

| File | Purpose |
| --- | --- |
| `asri_model.py` | Original ASRI head and 1-shot wrapper |
| `multishot.py` | Original 1-shot/5-shot wrapper |
| `controls.py` | Original input and branch ablations |
| `bilinear_control.py` | Original bilinear refinement control |
| `simple_head.py` | Original matched-input concatenation head |
| `metrics.py` | Original class-pooled regional and Boundary IoU metrics |
| `workflow.py` | Reusable head-training and paired evaluation helpers |
| `test_core.py` | CPU-only geometry, gradient, freezing and metric tests |
| `test_metrics.py` | Original boundary-metric unit tests |
| `protocol.json` | Paper settings and dataset sources |
| `SOURCE_MANIFEST.json` | Provenance and hashes of archived source files |

The six original implementation files above and the original metric tests are
copied without modification from the retained experiment sources. The workflow
helpers and integration tests were added for this release; they are not the
historical server scheduler.

## Installation and tests

Use Python 3.10 or newer for the standalone core package:

```bash
pip install -r requirements.txt
python -B -m unittest -v test_core test_metrics
```

These tests use a small synthetic base to check the wrapper contract. They do
not run SMNet, allocate a GPU, reproduce benchmark results, or create a new
scientific experiment. SMNet and its Mamba dependencies must be installed
separately for full-model use.

Release checks passed with PyTorch 2.7.1, NumPy 1.26.3 and SciPy 1.13.1,
using CPU execution with CUDA hidden. These are release-check versions, not
an assertion that this was the original server training environment.

## SMNet dependency and attribution

The base architecture is SMNet, not a new component claimed by this repository:

- Original implementation: https://github.com/Yanggzhao/SMNet
- Z. Yang et al., *SMNet: Self-support score-guided prototype mining and
  multi-scale pyramid Mamba for few-shot industrial defect segmentation*,
  Measurement 290 (2026), 122692.
  https://doi.org/10.1016/j.measurement.2026.122692

Obtain SMNet and its pretrained backbone from the original authors and follow
their dependency and use conditions. This package does not redistribute
third-party model source, pretrained weights or datasets. The tested project
carrier used project-specific initialization/freezing and training settings;
published SMNet checkpoints are not interchangeable with the project's final
weights when attempting to match numerical tables.

## Interface

The base model must expose `layer1` and accept
`base(support, support_mask, query, None)`, returning `(logits, auxiliary)`.
It must encode each support through `layer1`, followed by the query.

| Tensor | Shape |
| --- | --- |
| support | B x K x 3 x H x W |
| support_mask | B x K x H x W |
| query | B x 3 x H x W |
| base logits / output logits | B x 2 x H x W, channel 0 background |
| shallow query feature | B x 256 x hq x wq |

`K` is 1 or 5. H and W must be divisible by 4. Query features are aligned to
the quarter-resolution posterior grid before the 256-to-32 projection.

```python
from multishot import MultiShotASRI
from workflow import fit_head, evaluate_pair

# Build SMNet with the original implementation and load the desired final
# project base checkpoint strictly before constructing the wrapper.
base.load_state_dict(base_checkpoint["state_dict"], strict=True)
model = MultiShotASRI(base, mode="FREE").to(device)

# Each training batch is (query, target, support, support_mask, ...).
history = fit_head(model, train_loader, epochs=5)

# Each evaluation episode has batch size 1 and is
# (query, target, support, support_mask, fold_local_class_index).
results = evaluate_pair(model, episodes, class_names)
model.close()
```

`fit_head` does not choose data splits, construct episode streams, select the
best checkpoint or seed, or tune against test data. The caller supplies the
dataset-specific training loader and evaluation bank. Import any original data
loader modules that reset random seeds before setting the run seed; account for
the model constructor's RNG consumption when reproducing training streams.

For the simple-head control, replace the allocator with
`SimpleHead(model.allocator.query_projection)` before creating the optimizer.
`build_control(base, variant)` supplies the original 1-shot ablations:
`A1` shared bias, `A2` positional residual, `A3` no query details,
`A4` no posterior conditions, `A5` bilinear refinement, `A6` full ASRI.

## Parameterization

The main configuration is `FREE`: **raw positional residual + raw shared bias**.
The archived module also computes unused centered/gated diagnostic values.
They are not added to the main prediction. The allocator contains 28,563
parameters, of which 28,529 participate in the main FREE computation; 34 belong
to the unused diagnostic gate. These original fields are retained for source
and checkpoint compatibility. The simple head has 27,856 parameters.

## Protocol and availability

Paper settings are in `protocol.json`: ResNet-50, 200 x 200 input, three folds,
1-shot/5-shot evaluation, a fixed 201-epoch project base, and a five-epoch
AdamW refinement head. The one-parameter comparator adds a zero-initialized
trainable scalar to the frozen base margin; `GlobalOffset` implements this
operation in `workflow.py`.

Dataset origin pages:

- FSSD-12: https://github.com/VDT-2048/CPANet
- Surface Defects-4i / MSD-Seg2:
  https://github.com/bbbbby-99/TGRNet-Surface-Defect-Segmentation

`metrics.py` pools intersection/union counts over episodes within each class,
averages class IoUs within a fold, and uses a fixed 2% image-diagonal inner
boundary band. Call it per fold, then average the three fold summaries equally.
The width is six pixels for 200 x 200 inputs. No test-set tuning is performed.

This is a core-method code release, not a complete frozen artifact for replaying
every paper table. Final trained checkpoints, original episode banks, base
training scripts and server execution receipts are not bundled. Exact numeric
replay requires those original weights/streams and compatible dependencies.
The release helpers have been checked on CPU with a synthetic base; full SMNet
GPU integration has not been rerun for this release. No benchmark scores are
claimed from the synthetic tests.

## Rights

No open-source license is selected automatically. Public visibility is not a
grant of unrestricted reuse rights. Copyright and licensing decisions for the
authors' code remain with the rights holders; third-party dependencies retain
their own terms. Repository access is for inspection and review unless the
rights holders provide further permission.
