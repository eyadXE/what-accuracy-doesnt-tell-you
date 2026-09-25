# What Accuracy Doesn't Tell You

**How much information about a neural network's decision is lost as it moves down the pipeline:**

```
raw logits  →  softmax probabilities  →  predicted class (argmax)  →  accuracy
```

Each arrow above is a lossy, one-way compression. Softmax throws away the *overall level* of the
logits. Argmax throws away every class's score except the winner. Accuracy throws away everything
but one bit per sample, then averages it. This repo trains 7 models across 4 datasets, measures
that information loss directly, and writes it up.

📖 **Read the post:** [`blog/post.md`](blog/post.md) · also published at [eyadxe.github.io/blog.html](https://eyadxe.github.io/blog.html)
📓 **Full notebook:** [`notebooks/what_accuracy_doesnt_tell_you.ipynb`](notebooks/what_accuracy_doesnt_tell_you.ipynb)

This isn't a new method — the overconfidence/miscalibration result is from Guo et al. (2017), and
the energy-score angle is from Liu et al. (2020). This is an intuition-first, from-scratch,
hands-on walkthrough of those results, with every number pulled from a real run on a single
RTX 3060 — including the places where the results didn't go the way the papers predicted.

## Headline findings

- **Accuracy is exactly (bit-for-bit) invariant to positive logit scaling** (`z' = αz`, i.e.
  temperature scaling with `T = 1/α`) — verified across α ∈ {0.1 … 10} on every model. Confidence,
  NLL, and ECE all move dramatically at the same time.
- **Same accuracy, very different calibration:** two identically-architected MNIST MLPs differ by
  0.04 accuracy points (98.46% vs 98.42%) but the overtrained one has 67% higher ECE (0.0142 vs
  0.0085) and 2.6× the NLL (0.1506 vs 0.0573).
- **Bigger ≠ more trustworthy:** a CIFAR-10 ResNet-18 (86.7% accuracy) is 3× worse calibrated than
  a smaller SmallCNN (76.8% accuracy) on the same task — reproducing Guo et al.'s finding directly.
- **Softmax's shift-invariance is exact, not approximate:** shifting every logit by a constant
  changes softmax output by ≤ 4.77×10⁻⁷ (float32 noise) across a 10,000-sample test set, while
  `logit_mean` and `energy = -logsumexp(z)` shift by exactly that constant, every time.
- **A cross-validated logistic-regression probe on the full raw logit vector is the *worst* of
  five correctness-ranking scores on every model** — worse than the simple hand-derived margin or
  max-softmax probability. More raw information didn't mean a better score once you have to learn
  it from a handful of labeled examples.
- **Energy doesn't always beat softmax for OOD detection.** For one model (CIFAR-10 SmallCNN vs.
  Gaussian noise), *both* scores point the wrong way — the model is 93.2% confident that pure
  static is a dog, and its OOD-detection AUROC is 0.476, below chance.
- **`argmax_only` scores exactly 0.5 AUROC everywhere, by construction** — the cleanest possible
  demonstration that a single predicted class, with no accompanying score, carries zero
  information about which predictions to trust.

Full findings, all 15 bullets, and every table: see [`blog/post.md`](blog/post.md).

## Experiment grid

| Tag | Dataset | Architecture | Why |
|---|---|---|---|
| A | MNIST | MLP | Baseline, well-regularized |
| B | MNIST | MLP, overtrained (120 epochs, no regularization) | Same accuracy, worse calibration |
| C | Fashion-MNIST | SmallCNN | Harder grayscale task |
| D | CIFAR-10 | SmallCNN | Low capacity on a harder task |
| E | CIFAR-10 | ResNet-18 | Deeper, more accurate, expected more overconfident |
| F | CIFAR-100 | ResNet-18 | 100 classes — margin alone tells less of the story |
| G | CIFAR-10 | ResNet-18 + label smoothing | Same arch/data as E, different training choice |
| AUX3 | MNIST (digits 4/7/9) | MLP | Illustrative 3-class model for the geometry figures only |

Every model gets the same metric suite: accuracy, NLL, Brier score, ECE/MCE (15-bin, equal-width
and equal-mass), temperature scaling, margin/confidence correlation, and AUROC/AURC for both
misclassification detection and OOD detection (against Gaussian/uniform noise and 2-4 real OOD
datasets per model, shape/normalization-matched to the in-distribution model).

## Reproduce

```bash
conda create -n logits-blog python=3.11 -y
conda activate logits-blog
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/what_accuracy_doesnt_tell_you.ipynb
```

Model checkpoints and cached logits live under `artifacts/checkpoints/` and `artifacts/logits/`
(gitignored — not in this repo, regenerated on first run) — once trained, re-running the notebook
loads from cache instead of retraining. Delete a checkpoint file to force a retrain. The full run
(all 8 models, full metric suite, all figures) takes roughly 1.5-2 hours on an RTX 3060; set
`QUICK_MODE = True` in the notebook's `CONFIG` cell for a fast few-epoch smoke test instead.

## Layout

```
notebooks/            the story — training + all analysis (E1-E9), an interpretation cell
                       after every output cell, real numbers only
src/
  data.py              dataset loaders, ID + OOD, shape/normalization-matched
  models.py            MLP, SmallCNN, ResNet-18 (CIFAR variant)
  train.py             training loop with checkpoint caching
  metrics.py           ECE, MCE, NLL, Brier, AUROC, AURC, energy, temperature scaling
  plots.py             shared, colorblind-safe figure style
artifacts/tables/       exported result tables (CSV + Markdown): calibration, alpha-sweep,
                       margin/confidence, information-loss, master summary, training history
figures/               all 11 report figures + cover image, PNG (2x DPI) + SVG, CAPTIONS.md
blog/post.md            the published write-up
```

## References

- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On Calibration of Modern Neural
  Networks.* ICML.
- Liu, W., Wang, X., Owens, J., & Li, Y. (2020). *Energy-based Out-of-distribution Detection.*
  NeurIPS.
- Hendrycks, D., & Gimpel, K. (2017). *A Baseline for Detecting Misclassified and
  Out-of-Distribution Examples in Neural Networks.* ICLR.

## Author

Eyad Magdy Kamal — [portfolio](https://eyadxe.github.io) · [GitHub](https://github.com/eyadXE) ·
[LinkedIn](https://www.linkedin.com/in/eyad-magdy/)
