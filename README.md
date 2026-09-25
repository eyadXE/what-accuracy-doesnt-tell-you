# What Accuracy Doesn't Tell You

Research notebook and blog post about how much information about a neural network's
decision is lost as you move down the pipeline: `raw logits -> softmax probabilities ->
predicted class (argmax) -> accuracy`.

## Reproduce

```bash
conda create -n logits-blog python=3.11 -y
conda activate logits-blog
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/what_accuracy_doesnt_tell_you.ipynb
```

Model checkpoints and cached logits live under `artifacts/checkpoints/` and
`artifacts/logits/` — once trained, re-running the notebook loads from cache
instead of retraining. Delete a checkpoint file to force a retrain.

`QUICK_MODE` in the notebook's `CONFIG` dict switches to fewer epochs / smaller
subsets for fast iteration; set it to `False` for the full run used in the post.

## Layout

- `notebooks/` — the story: training + all analysis, with an interpretation cell
  after every output cell.
- `src/` — reusable code (`data.py`, `models.py`, `train.py`, `metrics.py`, `plots.py`).
- `artifacts/` — cached checkpoints/logits (gitignored) and exported result tables (CSV + Markdown).
- `figures/` — exported PNG/SVG figures for the blog, plus `CAPTIONS.md`.
- `blog/post.md` — the Hashnode-ready post.

## Summary

See `blog/post.md` once written, or the Phase 3 checkpoint summary for headline findings.

## References

- Guo et al. 2017, *On Calibration of Modern Neural Networks*.
- Liu et al. 2020, *Energy-based Out-of-distribution Detection*.
- Hendrycks & Gimpel 2017, baseline for misclassification/OOD detection via max softmax probability.
