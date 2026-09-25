# BLOG_TASKS.md: "What Accuracy Doesn't Tell You"

> **For Claude Code.** This file is the full brief for a blog project. Read it end to end before doing anything.
> Work through the phases **in order** and stop at every 🛑 checkpoint to wait for my input.
> The author is Eyad (AI/ML engineer). The blog is published on **Hashnode**.

---

## 0. The Big Picture

**Research question:** How much information about a neural network's decision is lost as we move down the pipeline

```
raw logits  →  softmax probabilities  →  predicted class (argmax)  →  accuracy
```

and how does this change the way we should read "confidence" and "correctness"?

**Core thesis:** Each arrow is a lossy compression.
- **softmax** throws away the *overall logit level*. It is invariant to adding a constant to every logit, so it cannot see the component of the logits along the all-ones direction.
- **argmax** throws away *everything except the ranking's winner*.
- **accuracy** throws away *everything except one bit per sample*, then averages it.

**Positioning (must be stated honestly in both the notebook and the post):**
- Overconfidence and miscalibration are known results: Guo et al. 2017, *On Calibration of Modern Neural Networks*.
- The α-scaling intervention `z' = αz` **is temperature scaling** with `T = 1/α`. Name it as such.
- The "what softmax discards" angle connects to energy-based OOD detection: Liu et al. 2020, *Energy-based Out-of-distribution Detection*, which uses the log-sum-exp of the logits.
- The contribution of this post is an **intuition-first, geometric, hands-on walkthrough**, not a new method.
- Do **not** call this "mechanistic interpretability." We only analyze the output layer.

---

## 1. Environment & Constraints

- Hardware: a single **RTX 3060 (12 GB)**. Every experiment must fit in memory and finish in a reasonable time. The full notebook should run in **≤ ~2–3 hours** total.
- Stack: Python 3.11+, PyTorch, torchvision, numpy, pandas, scikit-learn, matplotlib (seaborn optional), scipy, tqdm. Use `timm` only if needed for a small ViT.
- Fix all seeds (`torch`, `numpy`, `random`) and set `torch.backends.cudnn.deterministic = True` where practical.
- **Cache trained model checkpoints and logits to disk** (`artifacts/checkpoints/`, `artifacts/logits/*.npz`) so re-running analysis cells never retrains.
- Keep compute config at the top of the notebook in one `CONFIG` dict, including a `QUICK_MODE` flag with fewer epochs and subsets for fast iteration.

### Repo layout to create

```
logits-blog/
├── BLOG_TASKS.md                 # this file
├── README.md                     # how to reproduce, short summary
├── requirements.txt
├── notebooks/
│   └── what_accuracy_doesnt_tell_you.ipynb
├── src/
│   ├── data.py                   # dataset loaders (ID + OOD)
│   ├── models.py                 # MLP, SmallCNN, ResNet-18 (CIFAR variant), optional tiny ViT
│   ├── train.py                  # training loop, checkpointing
│   ├── metrics.py                # ECE, MCE, NLL, Brier, AUROC, AURC, entropy, energy...
│   └── plots.py                  # all figure functions, shared style
├── artifacts/
│   ├── checkpoints/
│   ├── logits/
│   └── tables/                   # CSV + markdown versions of result tables
├── figures/                      # exported charts for the blog (PNG + SVG)
└── blog/
    └── post.md                   # the Hashnode-ready post (Phase 4)
```

Keep reusable logic in `src/` and import it into the notebook. The notebook is for the **story**: running, showing, and explaining.

---

## 2. Phase 1: Experiments (the notebook)

### 2.1 The golden rule of this notebook

**Every cell that produces an output must be followed by a markdown cell that interprets that output.**

The interpretation cell must:
1. Say **what we are looking at** (axes, columns, units).
2. Report the **actual numbers** from *this run*, not generic expectations. Pull them from the output.
3. Explain **what it means** for the thesis (which information was preserved or lost).
4. Flag anything **surprising, contradictory, or weak**, honestly. A null result gets explained, not hidden.

Because interpretation depends on real outputs: **run the cell first, read the output, then write the explanation.** Never pre-write explanations. If a later re-run changes the numbers, update the text.

Each section also starts with a short markdown cell stating the **question** and the **prediction** (what we expect to see and why), so the reader can compare the prediction against reality.

### 2.2 Experimental grid

Vary across **datasets × architectures**, then analyze each resulting model with the **same metric suite**.

| Tag | In-distribution dataset | Architecture(s) | Why it's here |
|---|---|---|---|
| A | MNIST | MLP (2 hidden layers) | Easy task, baseline, probably well-calibrated |
| B | MNIST | MLP **overtrained** (e.g. 100+ epochs, no regularization) | Same accuracy, worse calibration: the thesis in one pair |
| C | Fashion-MNIST | SmallCNN | Harder grayscale task, more errors to study |
| D | CIFAR-10 | SmallCNN | Low capacity on a harder task |
| E | CIFAR-10 | ResNet-18 (CIFAR variant) | Modern, deep, expected to be overconfident (Guo et al.) |
| F | CIFAR-100 | ResNet-18 | Many classes, where the top-2 margin tells a less complete story |
| G *(optional)* | CIFAR-10 | tiny ViT or ResNet-18 trained **with label smoothing** | Shows how training choices reshape logit geometry |

**OOD datasets for the "what softmax discards" experiment:**
- MNIST models: Fashion-MNIST, KMNIST (or EMNIST letters), Gaussian noise, uniform noise
- CIFAR-10 models: SVHN, CIFAR-100 (near-OOD), Gaussian noise
- Match OOD input shape and normalization to the ID model's preprocessing.

Log per-model training curves (train/val loss + accuracy) and **val NLL alongside val accuracy**. Where NLL rises while accuracy stays flat, that is overfitting in confidence, and it is a key figure.

### 2.3 Per-sample record

For every test sample of every model, store in a DataFrame (and cache to `.npz`/parquet):

- `true_class`, `pred_class`, `correct`
- `logits` (full vector)
- `probs` (full softmax vector)
- `p_top1`, `p_top2`
- `margin = z_top1 − z_top2` (logit margin)
- `prob_margin = p_top1 − p_top2`
- `entropy` of softmax
- `energy = −logsumexp(z)` (report the sign convention clearly)
- `logit_norm = ‖z‖₂`
- `logit_mean = mean(z)`. This is the "softmax-invisible" component: the projection onto the all-ones direction.

### 2.4 Experiments

For each, produce outputs and interpretation cells as required by 2.1.

**E1. Same accuracy, different decision states**
Pick four archetypes per model: correct/large margin, correct/tiny margin, wrong/tiny margin, wrong/large margin. Show the input image, the logit bar chart, and the softmax bar chart side by side. The message: accuracy scores these as 1, 1, 0, 0, but they are very different situations.

**E2. The shift invariance of softmax (what softmax literally cannot see)**
- Add a constant `c` to all logits and show that softmax is unchanged, argmax is unchanged, and `logit_mean` and energy are changed.
- Show the distribution of `logit_mean` / energy across the test set. This is information that exists in the network and vanishes after softmax.

**E3. Temperature / α-scaling intervention**
- Sweep `α ∈ {0.1, 0.25, 0.5, 1, 2, 4, 10}` (plus a fine grid for plots), with `z' = αz`.
- Measure: accuracy (should be **identical**; verify exactly), predicted class agreement with α=1 (should be 100%), mean confidence, mean entropy, NLL, ECE, Brier.
- Fit the **optimal temperature** on a held-out validation split by minimizing NLL (proper temperature scaling), then report before/after on the test set.
- Explain: accuracy is blind to α, while every probabilistic metric is not. Hence accuracy alone cannot tell you whether your confidences mean anything.
- Also show the sanity case **α < 0** (it reverses the ranking) to make the point that only *positive* scaling preserves argmax.

**E4. Calibration: does 90% mean 90%?**
- Reliability diagrams (15 equal-width bins and also equal-mass bins) for every model, before and after temperature scaling.
- Show bin counts (a histogram under the diagram), since empty bins lie.
- Metrics table per model: Accuracy, ECE, MCE, adaptive ECE, NLL, Brier, mean confidence, and the "overconfidence gap" = mean confidence − accuracy.
- Key comparison: **A vs B** (same data and architecture, same-ish accuracy, different calibration) and **D vs E** (a bigger model is more accurate but more overconfident?).

**E5. Margin vs confidence as correctness predictors**
- Before running: derive and state that for the top two classes `p1/p2 = exp(z1 − z2)`, so max-softmax and margin are **strongly monotonically related**. Predict that they will perform similarly on some models.
- Plot empirical accuracy vs binned margin, and accuracy vs binned confidence.
- Compute Spearman correlation between margin and `p_top1`.
- Explain when they diverge: with many classes (CIFAR-100), probability mass spread over the tail makes `p_top1` differ from what the top-2 margin alone implies.

**E6. The information-loss table (the headline result)**
Quantify "how much is lost" at each stage. For each model, compute **AUROC** (and **AURC** from the risk-coverage curve) for two tasks:
1. **Misclassification detection:** separate correct from incorrect ID predictions.
2. **OOD detection:** separate ID test from each OOD set.

Using each score:

| Score | Pipeline stage |
|---|---|
| full logit vector (small logistic-regression probe, cross-validated) | raw logits |
| energy / logsumexp | raw logits (softmax-invisible part) |
| logit margin | raw logits |
| max softmax probability | softmax |
| negative entropy | softmax |
| argmax only (constant score, so AUROC = 0.5) | predicted class |
| — | accuracy (a single number; can't rank samples at all) |

Present this as one clean table (heatmap too) and export it. Explain which stages lose what, with the actual numbers. **If the numbers don't support the thesis for some model, say so and explain why.**

**E7. Confidently wrong**
- Top-k (e.g. 16) highest-confidence errors per model: image grid with true/pred labels, confidence, margin, energy.
- Discuss honestly: some are label noise or genuinely ambiguous images; others are real model failures.

**E8. Confident on garbage (the OOD punchline)**
- Feed noise and OOD datasets to each model.
- Show histograms of max-softmax for ID vs OOD (heavy overlap expected) next to histograms of energy for ID vs OOD (better separation expected).
- Pick a few funny examples, like "the model is 99.3% sure this static noise is a 7."

**E9. Cross-cutting summary**
One final section with a master table (model × {accuracy, ECE, NLL, Brier, AUROC-misclass by score, AUROC-OOD by score}) and a written summary of the findings across datasets and architectures: what generalizes, what doesn't, and what depends on capacity or training.

---

## 3. Phase 2: Geometry Charts for the Blog

Export **publication-quality figures** that explain the *geometry* of the problem. These are the hero images of the post.

### Export spec
- Save to `figures/` as **PNG (2x DPI, ~1600px wide)** and **SVG**.
- Filenames: `figNN_short_name.png` (e.g. `fig03_softmax_shift_invariance.png`).
- **Light background**, large readable fonts (≥ 14pt at export size), colorblind-safe palette, one consistent style defined in `src/plots.py`.
- No chart junk. Every figure must be readable on a phone.
- Also produce a **cover image, 1600×840**, for Hashnode: a clean, bold, geometric visual with the post title.
- Write `figures/CAPTIONS.md` with one line per figure: filename → caption → which blog section it belongs to → alt text.

### Required geometric figures

1. **The lossy pipeline diagram.** logits → softmax → argmax → accuracy, with what each arrow throws away annotated under it.
2. **3-class logit space.** Logits as points in ℝ³ with the all-ones direction drawn. Show that moving along it changes nothing after softmax: softmax "collapses" each line parallel to (1,1,1) to one point.
3. **The probability simplex (triangle).** Real test samples from a 3-class subset (or a 3-class model) plotted on the 2-simplex, colored by correct/incorrect. Show decision regions and the vertices where "overconfident" points pile up.
4. **Temperature as motion on the simplex.** Trajectories of a few samples as α goes 0 → ∞: they start at the center (uniform) and slide toward a vertex, while **never crossing a decision boundary**. The argmax invariance, visualized.
5. **Margin geometry in 2D.** The `z1 − z2` plane with the decision boundary as the line `z1 = z2`, with margin as the distance to it. Scatter real samples; show correct and wrong on both sides.
6. **The sigmoid of margin.** For 2 classes, `p = σ(margin)`. Plot it with real samples overlaid, showing why beyond a margin of ~5 everything reads as "≈100% confident."
7. **Reliability diagrams.** Before/after temperature scaling for the most overconfident model, with the gap shaded.
8. **Accuracy vs NLL over training.** For the overtrained model, accuracy flatlines while NLL climbs.
9. **ID vs OOD: softmax vs energy histograms.** Side by side.
10. **The information-loss heatmap.** From E6: scores × tasks, AUROC values.
11. **Confidently-wrong gallery.** Grid of images with labels and confidence.

Feel free to add figures if an experiment reveals something visual. Keep the total for the post to ~8–12 figures, and pick the best.

---

## 4. Phase 3: Review Checkpoint 🛑

Before writing any blog prose, **stop and send me**:
1. A short summary (≤ 15 bullets) of the actual findings, including any that contradict the thesis.
2. The E6 information-loss table and the master table from E9.
3. The list of exported figures with thumbnails or paths.
4. A proposed **outline** for the post (see Phase 4), with which figure goes where.
5. 3–5 title options.

**Wait for my feedback. Do not start drafting until I approve the outline.**

---

## 5. Phase 4: Writing the Blog Post Together

We write this **collaboratively, section by section**. Draft one section, show it to me, and incorporate my edits before moving on. Don't dump the whole post at once.

### 5.1 Tone & voice
- **Friendly, conversational, and genuinely funny**, like explaining this to a smart friend over coffee. Think self-aware jokes, playful analogies, the occasional dramatic reveal.
- Personify the model a bit ("the network is *extremely* sure this noise is a 7, and honestly, I admire the confidence").
- Humor must **never cost accuracy**. Every claim is backed by the notebook numbers. Jokes go around the science, not instead of it.
- Short paragraphs. Plenty of whitespace. One idea per paragraph.
- Assume the reader knows what a neural network and softmax are, but not calibration theory. Explain every metric the first time it appears, in one line.
- Written in the first person as Eyad. It should sound like a real person, not a textbook. Avoid AI-sounding filler ("delve," "in today's fast-paced world," "let's dive in," "it's important to note").

### 5.2 Suggested structure (to refine at the checkpoint)
1. **Hook:** two predictions, both "correct," both "confident," with wildly different logits. "Accuracy says these are identical. They are not."
2. **The pipeline and its three lossy arrows** (fig 1)
3. **What softmax can't see: the shift invariance** (fig 2)
4. **Turning the temperature knob: same accuracy, different truth** (figs 4, 6)
5. **Does 90% mean 90%? A calibration reality check** (figs 7, 8)
6. **Margin vs confidence: the plot twist that they're (mostly) the same thing**
7. **Confident on garbage: OOD and the energy that softmax threw away** (fig 9)
8. **The scoreboard: how much each stage loses** (fig 10)
9. **Hall of fame: the most confidently wrong predictions** (fig 11)
10. **So what? Practical takeaways** (e.g., log margins and energy, calibrate before trusting confidence, report NLL/ECE next to accuracy)
11. **Standing on shoulders:** short related-work credit (Guo et al. 2017; Liu et al. 2020; Hendrycks & Gimpel 2017 baseline for OOD/misclassification detection)
12. **Code & reproducibility:** link to the GitHub repo (github.com/eyadXE) and the notebook

### 5.3 Hashnode compatibility rules
Output the final post to `blog/post.md` following these rules:
- Plain **GitHub-flavored Markdown**. No HTML unless unavoidable. No Jupyter-specific syntax.
- **Do not** put the title as an `# H1` in the body. Hashnode takes the title from its own field. Start sections at `##`.
- At the very top, put a comment block (to be removed before publishing) listing: title, subtitle, suggested slug, 3–5 tags (e.g. `machine-learning`, `deep-learning`, `pytorch`, `data-science`, `artificial-intelligence`), a meta description (≤ 155 chars), and the cover image filename.
- **Math:** use display math blocks with `$$ ... $$` on their own lines. Keep inline math minimal, and verify it renders in Hashnode's preview. If inline doesn't render, rewrite it as display math or plain text / code (`z1 − z2`).
- **Images:** reference them as `![alt text](figures/figNN_name.png)` with a caption line in italics beneath. Add a note in the comment block that images must be uploaded to Hashnode and the paths replaced with Hashnode CDN URLs.
- **Code blocks** with language tags (```` ```python ````). Keep snippets short (≤ 15 lines) and focused on the key idea. The full code lives in the repo.
- Use `>` blockquotes for "TL;DR" and key-takeaway callouts.
- Target length: **~2,500–3,500 words** (≈ 12–15 min read). If it runs longer, propose splitting into a 2–3 part series instead of cramming.
- End with a friendly call to action (comments, follow, repo star), kept short and not salesy.

---

## 6. Definition of Done

- [ ] Notebook runs top to bottom from a clean kernel (with cached artifacts) without errors.
- [ ] Every output cell has an interpretation cell below it that uses the actual numbers.
- [ ] All grid models (A–F, G optional) are trained, cached, and analyzed with the full metric suite.
- [ ] Accuracy invariance under positive α is **verified exactly**, not just claimed.
- [ ] E6 information-loss table and E9 master table are exported to `artifacts/tables/` (CSV + markdown).
- [ ] All geometry figures are exported to `figures/` (PNG + SVG) with `CAPTIONS.md` and a cover image.
- [ ] Review checkpoint done and outline approved by me.
- [ ] `blog/post.md` is written collaboratively, Hashnode-compatible, with every number traceable to the notebook.
- [ ] README explains how to reproduce everything.

---

## 7. Ground Rules

- **Honesty over narrative.** If a result undercuts the thesis, it goes in the post, framed as a finding.
- **Numbers come from runs, never from memory.** If a number appears in text, it must exist in a notebook output or a table in `artifacts/tables/`.
- **Cite** the prior work named in section 0. Don't imply novelty that isn't there.
- **Ask me** when a decision changes the story (dropping a dataset, a surprising result, a structural change to the post).
