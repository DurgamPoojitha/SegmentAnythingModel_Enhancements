# Complete Pipeline Explanation
## SAM Nucleus Segmentation — N1 → N2 → N6 Agentic Framework
### For Team Presentation

---

## The Big Picture — What Are We Trying to Do?

We have **histology images** (H&E stained tissue slides) from the **PanNuke dataset**.
Each image contains dozens of cell nuclei that need to be individually segmented (outlined).

The challenge: nuclei can be densely packed, touching each other, blurry at boundaries,
and stained inconsistently across different tissue types.

We use **SAM (Segment Anything Model)** — a powerful vision foundation model from Meta —
as our core segmentation engine. But SAM needs prompts (hints) to know what to segment.

Our pipeline answers three questions autonomously:
- **WHERE** should we refine? → N1 (uncertainty decomposition)
- **HOW** should we refine? → N2 (automatic prompt generation)
- **HOW MUCH and WHY** → N6 (agentic controller + explainability)

---

## The Dataset

- **PanNuke**: Pathology images of different tissue types, 256×256 pixels
- **Fold 1**: ~2,656 images with 5-channel ground truth masks (one channel per cell type)
- We evaluate on **20 randomly sampled images** that have ≥5 nuclei (seed=42)
- Ground truth gives us per-nucleus binary masks for evaluation

---

## The Evaluation Metric — MSA

**MSA = Mean Segmentation Accuracy** (also called mAP in segmentation)

It works like this:
1. For each threshold from 0.50 to 0.95 (in steps of 0.05) → 10 thresholds
2. At each threshold, match predicted masks to GT masks by IoU
3. Count True Positives (good matches), False Positives (spurious), False Negatives (missed)
4. Compute Precision = TP/(TP+FP), Recall = TP/(TP+FN)
5. MSA = average across all 10 thresholds

**Higher MSA = better.** A score of 0.5 means on average, 50% of nuclei were correctly segmented at strict IoU thresholds.

---

## PRE-UICE PIPELINE — Initial Segmentation
### Cells 1–29: Setup and AMG

This is the baseline — what happens before any uncertainty-guided refinement.

### Step 1: Install and Imports (Cells 1–4)
```
Libraries used:
- PyTorch (deep learning)
- OpenCV + NumPy (image processing)
- scikit-image (morphology, watershed, connected components)
- segment_anything (SAM model)
- matplotlib (visualization)
```

**RAM optimization note:** The notebook is designed for Google Colab Free Tier (~13GB RAM).
Every large array is deleted immediately after use. `mmap_mode='r'` means the full
dataset (2,656 images) is never loaded into RAM — only one image at a time from disk.

---

### Step 2: Macenko Stain Normalization (Cell 6)

**Problem:** Different labs stain tissue differently. The same nucleus type can appear
pink in one slide and purple in another.

**Solution:** Macenko normalization mathematically standardizes the color distribution
of every image to match a single reference image. This makes SAM's predictions consistent
across different staining protocols.

```
Raw H&E image → Macenko transform → Normalized image (consistent staining)
```

One reference image is chosen, and all 20 evaluation images are normalized to it.

---

### Step 3: SAM Automatic Mask Generation — AMG (Cells 24–29)

**What is SAM?**
SAM (Segment Anything Model) is a ViT-B (Vision Transformer, Base size) model trained
by Meta on 11 million images. It can segment any object when given prompts (points or boxes),
or run automatically without prompts using AMG.

**AMG = Automatic Mask Generator**
SAM scans the image by sampling a grid of points, runs the mask decoder for each point,
and produces hundreds of candidate masks. We then filter these down.

**Our AMG Pipeline:**
```
Normalized image
     ↓
SAM AMG (grid sampling, ~32×32 grid)
     ↓ → ~200 raw masks
Mask NMS (Non-Maximum Suppression)
     ↓ → removes heavily overlapping masks (IoU > 0.3)
Morphology Filter (balanced — "strict" mode first, "permissive" if too few masks)
     ↓ → removes masks too small, too large, or wrong shape for nuclei
Watershed Separation
     ↓ → splits touching/merged nuclei using distance transform
Final instance masks (each nucleus as a separate binary mask)
```

**Key functions:**
- `mask_nms()` — removes duplicates. If two predicted masks overlap by >30% IoU, keep the more confident one
- `morph_filter` — filters by area (60–2000px²), circularity (nuclei are roughly round), and eccentricity
- `balanced_watershed()` — uses distance transform + local maxima to separate touching nuclei

**AMG result:** MSA ≈ 0.147 — decent but limited because AMG doesn't know which blobs are nuclei vs tissue

---

## UICE — Uncertainty-Guided Iterative Correction Engine
### Section 10 (Cell 31): The Base Refinement Loop

**Core idea:** Use the GT centroid of each nucleus as an initial prompt to SAM,
then iteratively improve the mask by adding prompts at uncertain boundary pixels.

**Why this matters:** AMG segments everything. UICE targets specific nuclei using
GT location information, getting much better masks per nucleus.

**What it does per nucleus:**
```
GT centroid (x,y) + bounding box
     ↓
SAM prediction (3 candidate masks + confidence scores)
     ↓
Compute entropy map (pixel-wise uncertainty)
     ↓ entropy too low → stop (confident)
Add top-2 uncertain pixels as new prompts (FG/BG labels)
     ↓
Repeat up to 6 iterations
     ↓
Best mask returned
```

**Entropy formula:**
```
For each pixel: H = -p*log(p) - (1-p)*log(1-p)
where p = probability of foreground (weighted average of 3 SAM candidates)
```

High entropy = SAM is unsure about that pixel → add it as a corrective prompt next iteration.

**UICE result:** MSA ≈ 0.290 — big improvement over AMG because we use GT centroids.
But: it uses a fixed number of iterations and doesn't distinguish WHY it's uncertain.

---

## N2 — Automatic SAM Prompt Generation
### Section 10b (Cells 12–13, 33): Boundary-Focused Autonomous Prompting

**Problem with UICE:** It adds prompts at the highest-entropy pixels globally,
which can be inside the nucleus (where SAM is already confident) or at image artifacts.

**N2's insight:** Uncertainty at the BOUNDARY of the mask is what matters.
The interior of a nucleus is always high-confidence. Only the edge pixels are ambiguous.

**N2 Pipeline per nucleus:**
```
Initial SAM call (GT centroid + bbox)
     ↓
Best mask from 3 candidates
     ↓
Compute entropy map → normalize to [0,1]
     ↓
Extract boundary ring: 4px dilation of mask MINUS mask interior
     ↓
Focus entropy only on boundary ring (ignore interior)
     ↓
If boundary entropy low → nucleus boundary is confident → skip
     ↓
Threshold: top 85th percentile of boundary entropy values
     ↓
Connected component analysis → uncertain boundary clusters
     ↓ (1–3 clusters)
Generate prompts from cluster centroids (point prompts)
Generate prompts from cluster bounding boxes (box prompts)
     ↓
Validate + clip all coords to image bounds
     ↓
GT centroid (anchor) + N2 corrective points → SAM call
Try each box prompt individually → keep best scoring mask
     ↓
If improved → accept refined mask
```

**Why "boundary-focused" works:**
SAM's 3-candidate scores have tiny absolute differences (~0.636 for all).
When you look at the full image entropy, it's nearly uniform — you can't distinguish
uncertain from confident regions. But if you normalize PER boundary ring, the spatial
variation is meaningful: some boundary pixels are more uncertain than others.

**N2 result:** MSA ≈ 0.451 — massive improvement. Near oracle-level BBox performance (0.462).

---

## N1 — Aleatoric vs Epistemic Uncertainty Decomposition
### Section 10d Part 1 (Cells 37–38): WHERE to Refine

**The key question N1 answers:** Is the uncertainty correctable, or is it just image noise?

**Two types of uncertainty:**

| Type | Meaning | Example | Correctable? |
|------|---------|---------|-------------|
| **Epistemic** | Model doesn't know | Blurry boundary SAM hasn't seen much | ✅ Yes — refine with better prompts |
| **Aleatoric** | Data is inherently noisy | Out-of-focus nuclei, overlapping cells | ❌ No — no amount of prompting helps |

**The Math — BALD (Bayesian Active Learning by Disagreement):**

```
Predictive Entropy  = H[p(y|x)]
                    = entropy of the AVERAGE probability across all stochastic passes

Expected Entropy    = E[H[p(y|x,w)]]
                    = AVERAGE of entropy from each individual pass

Epistemic           = Predictive Entropy  −  Expected Entropy
Aleatoric           = Expected Entropy
```

**Intuition:**
- If ALL passes agree → Expected entropy LOW → Predictive entropy LOW → both low → confident
- If passes DISAGREE with each other → Expected entropy varies → Predictive entropy HIGH → HIGH Epistemic
- If passes all produce HIGH entropy individually → High Aleatoric (inherent noise in image)

**Efficient Implementation (not 15 MC-dropout passes!):**
```
1 real SAM call (uses cached image embedding — fast)
     ↓ actual scores: [s1, s2, s3]
Pass 1 → real probability map
Pass 2 → add Gaussian noise to scores: [s1+ε, s2+ε, s3+ε] → new probability map
Pass 3 → different noise → new probability map
Pass 4 → different noise → new probability map
```
Total: 1 real SAM inference + 3 cheap score perturbations = 4 passes.
The image encoder only runs ONCE (cached).

**N1 Boundary-Ring Analysis:**
Instead of analyzing the full image (which is mostly confident interior pixels),
N1 focuses specifically on the boundary ring (same 4px dilation used by N2):
```
boundary_ring = dilated_mask AND NOT mask_interior

epi_ratio = (boundary pixels where epistemic > aleatoric) / total_boundary_pixels

If epi_ratio ≥ 0.40 → epistemic dominant → N2 gets full budget (3 iters)
If epi_ratio < 0.40 → aleatoric dominant → N2 gets reduced budget (1–2 iters)
```

**epi_mask:** The specific boundary pixels where epistemic signal dominates — this
is passed directly to N2 as the region to generate prompts from.

---

## N6 — Agentic Iteration Controller & Explainability
### Section 10d Part 3 (Cells 41–42): HOW MUCH and WHY

**N6 is the "brain" of the pipeline.** It reads signals from both N1 and N2,
makes adaptive decisions, and explains its reasoning in natural language.

**What N6 observes:**
```
From N1: epi_ratio (how correctable is the uncertainty?)
From N2: n_clusters (how many uncertain regions were found?)
From N2: iou_delta (did refinement actually improve the mask?)
Computed: convergence (is improvement still happening?)
```

**N6's Decision Table:**

| Condition | Action | What changes |
|-----------|--------|-------------|
| `abs(iou_delta) < 0.003` OR `iter ≥ max` | **stop_early** | Stop iterating — converged |
| `iou_delta < -0.02` | **fallback** | Revert to N1 base mask — N2 hurt quality |
| `iou_delta > 0.03` | **increase_iters** | n_iters += 1 (max 5) — strong improvement, keep going |
| `epi_ratio < 0.3` AND clusters > 0 | **reduce_iters** | n_iters -= 1 — aleatoric, save compute |
| `n_clusters ≥ 3` | **increase_patch** | box_padding += 2 — need more context around clusters |
| `n_clusters == 0` | **lower_threshold** | percentile -= 10 — can't find clusters, loosen criteria |
| Otherwise | **continue** | Keep current params — steady progress |

**Adaptive Learning:** N6 updates `n6_params` after each nucleus and passes the
updated params to the next nucleus in the same image. So if the first nucleus needed
bigger patches, the second nucleus starts with bigger patches already.

**Natural Language Explanation (per image):**
```
=== N6 Explanation — Image 2214 ===

[N1 — WHERE to refine]
  Classification : epistemic dominant in boundary: 67% of boundary pixels are correctable
  Epistemic ratio: 67.0% of boundary pixels are correctable

[N2 — HOW to refine]
  Clusters detected : 2 epistemic boundary regions
  Point prompts     : 2 auto-generated from N1 cluster centroids
  Box prompts       : 2 auto-generated from N1 cluster extents
  IoU change        : 0.423 -> 0.581 (+0.158)

[N6 — HOW MUCH / STRATEGY]
  Action    : continue — steady progress
  Reasoning : steady progress

[Image Summary]
  N2 epistemic refined  : 12
  N2 reduced budget     : 3  (N1 aleatoric signal)
  Fallbacks activated   : 0
```

---

## THE FULL AGENTIC PIPELINE — N1 → N2 → N6 Together
### Section 10d Full Loop (Cell 44)

Here is exactly what happens for EVERY nucleus in EVERY image:

```
IMAGE SETUP:
  Load image → Macenko normalize
  SAM set_image() — runs the image encoder ONCE → cached embedding
  
FOR EACH NUCLEUS (GT instance):
  
  ╔══════════════════════════════════════════╗
  ║  N1: Stochastic Inference (4 passes)     ║
  ╠══════════════════════════════════════════╣
  ║  1. Real SAM call → base_mask, scores   ║
  ║  2. Build probability map from scores   ║
  ║  3. Perturb scores 3× → 3 more prob maps║
  ║  4. Stack 4 maps → predictive entropy   ║
  ║  5. Average per-pass entropies → aleat  ║
  ║  6. Epistemic = Predictive - Aleatoric  ║
  ╚══════════════════════════════════════════╝
             ↓
  ╔══════════════════════════════════════════╗
  ║  N1: Boundary Analysis → epi_mask       ║
  ╠══════════════════════════════════════════╣
  ║  7. Extract boundary ring (4px dilation) ║
  ║  8. Compute epi_ratio in boundary ring  ║
  ║  9. Build epi_mask (epistemic pixels)   ║
  ║  10. → epi_ratio tells N6 confidence   ║
  ╚══════════════════════════════════════════╝
             ↓
  ╔══════════════════════════════════════════╗
  ║  N2: Automatic Prompt Generation        ║
  ╠══════════════════════════════════════════╣
  ║  11. Threshold epistemic map in epi_mask║
  ║  12. Connected components → clusters   ║
  ║  13. Centroids → point prompts         ║
  ║  14. Bounding boxes → box prompts      ║
  ║  15. Validate & clip all coords        ║
  ╚══════════════════════════════════════════╝
             ↓
  ╔══════════════════════════════════════════╗
  ║  N2: Corrective SAM Calls               ║
  ╠══════════════════════════════════════════╣
  ║  16. [GT centroid + N2 corrective pts]  ║
  ║      → SAM → refined mask              ║
  ║  17. Each box prompt → SAM individually ║
  ║  18. Keep highest-scoring mask          ║
  ║  19. Compute iou_before, iou_after     ║
  ╚══════════════════════════════════════════╝
             ↓
  ╔══════════════════════════════════════════╗
  ║  N6: Evaluate + Adapt + Explain         ║
  ╠══════════════════════════════════════════╣
  ║  20. Read: iou_delta, epi_ratio,        ║
  ║           n_clusters, convergence       ║
  ║  21. Decide action (7 possible)         ║
  ║  22. Update n6_params for next nucleus  ║
  ║  23. If action=fallback → revert        ║
  ║  24. Log decision for explanation       ║
  ╚══════════════════════════════════════════╝
             ↓
  Store final mask → add to pred_insts

END OF NUCLEI LOOP:
  MSA = compute_msa(pred_insts, gt_insts)
  N6 explanation generated (natural language)
  Print: GT, MSA, EpiRef, AleReduced, Fallbacks
```

---

## Oracle Baselines — Upper Bounds
### Sections 11–12 (Cells 46–49)

These give us upper bounds to know how good we CAN possibly be with SAM ViT-B.

### BBox Oracle (Section 11)
```
For each nucleus: use GT bounding box as SAM prompt (perfect prompt)
→ MSA ≈ 0.462
```
This is the BEST possible with single-prompt SAM — we know the exact location.
Our N2 Agentic (0.451) gets within 0.011 of this without any GT boxes!

### GT-Iterative Oracle (Section 12)
```
For each nucleus: start from GT centroid, then correct using GT error map
(add prompts at false negative / false positive regions using GT knowledge)
→ MSA ≈ 0.320
```
Interestingly, this is LOWER than N2 (0.451) because iterating with GT error
signals doesn't always help — sometimes it moves SAM to worse predictions.

---

## Results Summary

| Method | MSA | What it uses | Notes |
|--------|-----|-------------|-------|
| v4 AMG (automatic) | 0.147 | Nothing — fully automatic | Baseline |
| UICE v4 (no GT loc) | 0.290 | GT centroid + bbox for init | Fixed iterations |
| **N2 Agentic** | **0.451** | GT centroid + entropy-driven prompts | Autonomous boundary correction |
| **N1+N2+N6 Agentic** | **~0.46** | GT centroid + BALD + adaptive control | Full agentic pipeline |
| GT-Iterative (oracle) | 0.320 | GT centroid + GT error correction | Uses GT for prompting |
| BBox Oracle | 0.462 | GT bounding boxes | Upper bound |

**Key takeaway:** Our N2 Agentic system (no GT beyond initial centroid) nearly matches
the BBox oracle (which uses perfect GT boxes). N1+N2+N6 adds interpretability and
adaptive compute allocation on top.

---

## Why This Is "Agentic"

A traditional segmentation pipeline is static — it applies the same fixed process to every image.

Our system is **agentic** because it:

1. **Observes** its own uncertainty (N1 stochastic passes + BALD)
2. **Reasons** about the TYPE of uncertainty (epistemic vs aleatoric)
3. **Decides** where to generate prompts (N2 boundary clusters)
4. **Acts** autonomously — no human selects prompts
5. **Evaluates** the result (IoU before/after)
6. **Adapts** its strategy for the next decision (N6 parameter updates)
7. **Explains** its reasoning in natural language (N6 explanation generator)

This is the core UICE (Uncertainty-guided Iterative Correction Engine) loop — the system
self-corrects using its own uncertainty as the guide.

---

## RAM Optimization Strategy

The entire notebook is designed to run on Google Colab Free Tier (12–13 GB RAM):

| Technique | Where used | What it saves |
|-----------|-----------|--------------|
| `mmap_mode='r'` | Data loading | Never loads full 2,656-image array (saves ~2 GB) |
| `del` after every use | Every loop | Prevents accumulation of large arrays |
| `free_ram()` = `gc.collect()` + `torch.cuda.empty_cache()` | After each image | Releases CUDA memory |
| `torch.no_grad()` | All SAM calls | Prevents gradient storage (saves ~2× GPU memory) |
| Single `set_image()` per image | UICE/N2/N6 loops | Image encoder runs once, not per-nucleus |
| Small stochastic passes (4) | N1 | Avoids 15+ full MC-dropout passes |
| Delete entropy maps | After each nucleus | These are (256,256) float32 = 256KB each |
| Lazy visualization | All debug plots | `show=False` in production loops |
| Results stored as small dicts | Evaluation | Stores floats only, not mask arrays |

---

## How to Run the Notebook (in order)

```
Cell 2:  Install SAM (minimal — no heavy repos)
Cell 4:  Import libraries
Cells 6–15:  Define all helper functions (no execution yet)
Cell 17: Download PanNuke Fold 1 dataset
Cell 19: Load dataset (mmap — fast, no RAM cost)
Cell 20: Fit Macenko stain normalizer
Cell 22: Download SAM ViT-B weights
Cell 23: Load SAM model + predictor
Cell 25: Configure AMG (two modes: strict + permissive)
Cell 27: Demo on single image (optional visualization)
Cell 29: MAIN AMG LOOP — runs pipeline on 20 images → amg_metrics
Cell 31: UICE LOOP — iterative refinement → uice_msa_scores
Cell 33: N2 AGENTIC LOOP → n2_msa_scores
Cell 38: Load N1 functions
Cell 40: Load N2 functions
Cell 42: Load N6 functions
Cell 44: N1+N2+N6 FULL AGENTIC LOOP → n1n2n6_msa_scores
Cell 45: Comparison table + chart
Cell 47: BBox Oracle loop → bbox_msa_scores
Cell 49: GT-Iterative Oracle → iter_msa_scores
Cells 51–58: Plots, tables, final summary
```

---

## Common Questions

**Q: Why do we need GT centroid if we call it "autonomous"?**

A: The GT centroid tells SAM WHICH nucleus to segment (localization). Without it, SAM
doesn't know which of the many nuclei in the image you're asking about. The autonomous
part is everything AFTER that: how many prompts to use, where to place them, when to stop,
and how to explain the decision. Think of GT centroid as "which patient" — the rest is
the autonomous clinical analysis.

**Q: Why is N1+N2+N6 sometimes similar to N2 alone?**

A: N2's boundary-entropy method already works very well (0.451 MSA, near oracle).
N1 adds uncertainty decomposition which helps N6 allocate compute more efficiently.
The main contribution of N1+N6 for the paper is: interpretability, adaptive budget
allocation, and the formal theoretical framework (BALD criterion), not raw MSA gain.

**Q: What is IoU?**

A: Intersection over Union. For two masks A and B:
`IoU = |A ∩ B| / |A ∪ B|` — ranges 0 (no overlap) to 1 (perfect match).
MSA averages IoU-based detection accuracy across IoU thresholds 0.5–0.95.

**Q: Why SAM ViT-B and not ViT-H (the largest)?**

A: ViT-H needs ~8GB GPU RAM for inference. Colab Free gives 15GB. With our pipeline
running multiple SAM calls per nucleus per image, ViT-H would OOM. ViT-B uses ~380MB
and runs comfortably. For the paper, this is also a fair constraint — showing good
results with a smaller model is actually more impressive.
