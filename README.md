# Interpretable Semantic Gradients in SSD: A PCA Sweep Approach and a Case Study on AI Discourse

Code and results for the paper accepted to **Findings of ACL 2026**.

## Abstract

Supervised Semantic Differential (SSD) is a mixed quantitative–interpretive method that models how text meaning varies with continuous individual-difference variables by estimating a semantic gradient in an embedding space and interpreting its poles through clustering and text retrieval. SSD applies PCA before regression, but currently no systematic method exists for choosing the number of retained components, introducing avoidable researcher degrees of freedom in the analysis pipeline. We propose a *PCA sweep* procedure that treats dimensionality selection as a joint criterion over representation capacity, gradient interpretability, and stability across nearby values of K. We illustrate the method on a corpus of short posts about artificial intelligence written by Prolific participants who also completed Admiration and Rivalry narcissism scales. The sweep yields a stable, interpretable Admiration-related gradient contrasting optimistic, collaborative framings of AI with distrustful and derisive discourse, while no robust alignment emerges for Rivalry. We also show that a counterfactual using a high-PCA dimension solution heuristic produces diffuse, weakly structured clusters instead, reinforcing the value of the sweep-based choice of K. The case study shows how the PCA sweep constrains researcher degrees of freedom while preserving SSD's interpretive aims, supporting transparent and psychologically meaningful analyses of connotative meaning.

---

## Repository Structure

```
.
├── main.py                  # Main analysis script (Dolma 2024 embeddings)
├── camera_ready.py          # Camera-ready replication script (GloVe 840B embeddings)
├── pca_sweep.py             # PCA sweep procedure implementation
├── requirements.txt         # Python dependencies
│
├── data/
│   └── AI_POSTS.xlsx        # Short AI-topic posts from Prolific + ADM/RIV narcissism scores
│
├── results/
│   ├── dolma_2024/          # Results using Dolma 2024 embeddings
│   │   ├── sweep/
│   │   │   ├── ADM/         # Sweep-selected K results for Admiration
│   │   │   └── RIV/         # Sweep-selected K results for Rivalry
│   │   ├── 120/
│   │   │   ├── ADM/         # High-K (K=120) counterfactual for Admiration
│   │   │   └── RIV/         # High-K (K=120) counterfactual for Rivalry
│   │   ├── cluster_comparison_dolma_2024_ai_ADM.png  # Sweep vs K=120 cluster comparison (ADM)
│   │   └── cluster_comparison_dolma_2024_ai_RIV.png  # Sweep vs K=120 cluster comparison (RIV)
│   │
│   └── glove.840B.300d/     # Same structure replicated with GloVe 840B embeddings
│       ├── sweep/
│       │   ├── ADM/
│       │   └── RIV/
│       └── 120/
│           ├── ADM/
│           └── RIV/
│
└── shed/
    └── ssd_lite_main.py     # Lightweight standalone SSD script (development scratch)
```

### Output files (per trait, per embedding model)

Each results subfolder contains:

| File | Description |
|------|-------------|
| `ssd_stats_summary_*.xlsx` | Regression statistics: R², F, p, β norm, effect sizes, % variance explained |
| `ssd_scores_*.xlsx` | Per-document SSD scores with original text |
| `cluster_members_*.xlsx` | All neighbor words with their cluster assignments |
| `clusters_pos_*.xlsx` | Positive-pole clusters: size, coherence, top words, representative excerpt |
| `clusters_neg_*.xlsx` | Negative-pole clusters: same columns |
| `*_snippets_pos_cluster_N.xlsx` | Representative text snippets for positive cluster N |
| `*_snippets_neg_cluster_N.xlsx` | Representative text snippets for negative cluster N |
| `*_pca_k_joint_auck_ONEPLOT.png` | Sweep diagnostic plot: interpretability AUCK and beta stability vs. K |

---

## Scripts

### `pca_sweep.py`

Implements the PCA sweep procedure. For each candidate K, it fits an SSD model and records:
- **Interpretability aggregate**: product of weighted mean cluster coherence and mean cosine-to-beta, detrended by log(variance explained) and smoothed via AUCK (Area Under the Curve of K-neighborhood).
- **Gradient stability**: 1 − cosine similarity between consecutive beta unit vectors, negated so higher is better, then smoothed via AUCK.

The **joint score** is the mean of both AUCK signals; the smallest K achieving the maximum joint score is selected.

### `main.py`

Runs the full analysis pipeline with **Dolma 2024** embeddings (`dolma_300_2024_1.2M.100_combined.kv`):
1. Loads data and NLP resources.
2. Runs the PCA sweep over K ∈ {1, 3, 5, …, 119} for each trait (ADM, RIV).
3. Fits the final SSD at the sweep-selected K and saves all result tables to `results/dolma_2024/sweep/<trait>/`.
4. Fits a counterfactual SSD at K=120 and saves to `results/dolma_2024/120/<trait>/`.
5. Saves a two-panel cluster comparison figure (sweep K vs. K=120).

### `camera_ready.py`

Identical pipeline using **GloVe 840B** embeddings (`glove.840B.300d.kv`). Outputs go to `results/glove.840B.300d/`.

---

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**External embedding files** (not included in this repository due to file size):

| Script | Embedding | Source |
|--------|-----------|--------|
| `main.py` | Dolma 2024 300d | [GloVe project](https://nlp.stanford.edu/projects/glove/) |
| `camera_ready.py` | GloVe 840B 300d | [GloVe project](https://nlp.stanford.edu/projects/glove/) |

Update the `MODEL_PATH` variable at the top of each script to point to your local copy of the embedding file (`.kv` format, loadable with `gensim`).

The SSD implementation is provided by the [`ssdiff`](https://pypi.org/project/ssdiff/) package (v0.1.5).

---

## Citation

If you use this code or data, please cite:

```
@misc{plisiecki_interpretable_2026,
	title = {Interpretable {Semantic} {Gradients} in {SSD}: {A} {PCA} {Sweep} {Approach} and a {Case} {Study} on {AI} {Discourse}},
	copyright = {Creative Commons Attribution 4.0 International},
	shorttitle = {Interpretable {Semantic} {Gradients} in {SSD}},
	url = {https://arxiv.org/abs/2603.13038},
	doi = {10.48550/ARXIV.2603.13038},
	abstract = {Supervised Semantic Differential (SSD) is a mixed quantitative-interpretive method that models how text meaning varies with continuous individual-difference variables by estimating a semantic gradient in an embedding space and interpreting its poles through clustering and text retrieval. SSD applies PCA before regression, but currently no systematic method exists for choosing the number of retained components, introducing avoidable researcher degrees of freedom in the analysis pipeline. We propose a PCA sweep procedure that treats dimensionality selection as a joint criterion over representation capacity, gradient interpretability, and stability across nearby values of K. We illustrate the method on a corpus of short posts about artificial intelligence written by Prolific participants who also completed Admiration and Rivalry narcissism scales. The sweep yields a stable, interpretable Admiration-related gradient contrasting optimistic, collaborative framings of AI with distrustful and derisive discourse, while no robust alignment emerges for Rivalry. We also show that a counterfactual using a high-PCA dimension solution heuristic produces diffuse, weakly structured clusters instead, reinforcing the value of the sweep-based choice of K. The case study shows how the PCA sweep constrains researcher degrees of freedom while preserving SSD's interpretive aims, supporting transparent and psychologically meaningful analyses of connotative meaning.},
	urldate = {2026-03-16},
	publisher = {arXiv},
	author = {Plisiecki, Hubert and Leniarska, Maria and Piotrowski, Jan and Zajenkowski, Marcin},
	year = {2026},
	note = {Version Number: 1},
	keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences},
}
```
