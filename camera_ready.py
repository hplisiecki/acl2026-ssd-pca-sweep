import os
import re
import pandas as pd


from ssdiff import (
    SSD,
    load_embeddings,
    normalize_kv,
    load_spacy,
    load_stopwords,
    preprocess_texts,
    build_docs_from_preprocessed,
)

from pca_sweep import pca_sweep

# -----------------------
# CONFIG
# -----------------------
DF_PATH = r"data\AI_POSTS.xlsx"
RESULTS_DIR = r"results"

# Use only Dolma embeddings
MODEL_NAME = "glove.840B.300d"
MODEL_PATH = r"D:\Resources\NLP\EN\glove.840B.300d\glove.840B.300d.kv"

# Apply only to AI texts
SLICE_NAME = "ai"          # kept only for printing
TEXT_COL = "text_ai"

TRAITS = {
    "ADM": "ADM",
    "RIV": "RIV",
}

# Sweep grid
PCA_K_VALUES = list(range(1, 121, 2))

# SSD settings
SSD_LEXICON = None
SSD_USE_FULLDOC = True
SSD_SIF_A = 1e-3

# Sweep settings
SWEEP_CLUSTER_TOPN = 100
SWEEP_K_MIN = 2
SWEEP_K_MAX = 5
SWEEP_TOP_WORDS = 20
SWEEP_WEIGHT_BY_SIZE = True

# Robustness hypers (used inside pca_sweep scoring)
AUCK_RADIUS = 3
BETA_SMOOTH_WIN = 7
BETA_SMOOTH_KIND = "median"

# What to save from sweep
SAVE_SWEEP_TABLE = False   # saves df_joined to xlsx
SAVE_SWEEP_FIGURE = True   # keep False to avoid plots entirely

# Final clustering params (full run)
FULL_CLUSTER_PARAMS = dict(
    topn=100,
    k=None,
    k_min=2,
    k_max=5,
    verbose=True,
    top_words=20,
)

# -----------------------
# LOAD DATA + NLP ONCE
# -----------------------
df_all = pd.read_excel(DF_PATH)

# drop nans in TEXT_COL and TRAITS
df_all = df_all.dropna(subset=[TEXT_COL] + list(TRAITS.values())).reset_index(drop=True)

df_all['word_count'] = df_all[TEXT_COL].apply(lambda x: len(str(x).split()) if pd.notnull(x) else 0)

print(f"Mean word count for {SLICE_NAME} texts: {df_all['word_count'].mean():.2f}")

os.makedirs(RESULTS_DIR, exist_ok=True)

print("Loading NLP resources once...")
nlp = load_spacy("en_core_web_sm")
stopwords = load_stopwords("en")


# -----------------------
# HELPERS (final SSD outputs)
# -----------------------

_ILLEGAL_RE = re.compile(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]')

def sanitize_df_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove characters that Excel (openpyxl) doesn't accept from all string cells.
    """
    def _clean(x):
        if isinstance(x, str):
            return _ILLEGAL_RE.sub("", x)
        return x
    return df.applymap(_clean)


def save_snippets_by_cluster(df_snip, out_dir: str, prefix: str):
    if df_snip is None or len(df_snip) == 0:
        return
    for cluster in df_snip["centroid_label"].unique():
        snippets = df_snip[df_snip["centroid_label"] == cluster].copy()
        snippets = snippets[["cosine", "snippet_anchor"]]
        snippets.columns = ["Cosine Similarity", "Snippet Anchor"]

        # sanitize before saving
        snippets = sanitize_df_for_excel(snippets)

        snippets.to_excel(
            os.path.join(out_dir, f"{prefix}_snippets_{cluster}.xlsx"),
            index=False,
        )


def build_representative_snippets(df_clusters, df_pos_snip, df_neg_snip):
    if df_clusters is None or len(df_clusters) == 0:
        return df_clusters
    df_snip = pd.concat([df_pos_snip, df_neg_snip], ignore_index=True)
    reps = []
    for _, row in df_clusters.iterrows():
        cluster_id = row["cluster_rank"]
        sign = row["side"]
        centroid_label = f"{sign}_cluster_{cluster_id}"
        sub = df_snip[df_snip["centroid_label"] == centroid_label]
        reps.append("" if len(sub) == 0 else sub.iloc[0]["snippet_anchor"])
    df_clusters = df_clusters.copy()
    df_clusters["representative_snippet"] = reps
    return df_clusters


def format_clusters_tables(df_clusters):
    pos_clusters = df_clusters[df_clusters["side"] == "pos"].copy()
    neg_clusters = df_clusters[df_clusters["side"] == "neg"].copy()

    for d in (pos_clusters, neg_clusters):
        d.drop(
            columns=[c for c in ["cluster_rank", "side"] if c in d.columns],
            inplace=True,
        )

    cols = ["Size", "Cos β", "Coherence", "Top Words", "Representative Excerpt"]
    pos_clusters.columns = cols
    neg_clusters.columns = cols

    pos_clusters["No."] = range(1, len(pos_clusters) + 1)
    neg_clusters["No."] = range(1, len(neg_clusters) + 1)

    cols_out = ["No.", "Size", "Cos β", "Coherence", "Top Words", "Representative Excerpt"]
    return pos_clusters[cols_out], neg_clusters[cols_out]


def run_and_save_final_ssd(*, kv, docs, pre_docs, texts, y, trait_name: str, pca_k: int, out_dir: str, prefix: str):
    ssd = SSD(
        kv,
        docs,
        y,
        lexicon=SSD_LEXICON,
        use_full_doc=SSD_USE_FULLDOC,
        N_PCA=int(pca_k),
        SIF_a=SSD_SIF_A,
    )

    # ---- stats
    stats = {
        "Trait": trait_name,
        "PCA_K": int(pca_k),
        "R2": ssd.r2,
        "adj_R2": float(getattr(ssd, "r2_adj", float("nan"))),
        "F": ssd.f_stat,
        "p": ssd.f_pvalue,
        "beta_norm": ssd.beta_norm_stdCN,
        "delta_per_0.10_raw": ssd.delta_per_0p10_raw,
        "IQR_effect_raw": ssd.iqr_effect_raw,
        "corr_y_pred": ssd.y_corr_pred,
        "n_raw": int(getattr(ssd, "n_raw", len(docs))),
        "n_kept": int(getattr(ssd, "n_kept", len(docs))),
        "n_dropped": int(getattr(ssd, "n_dropped", 0)),
        "prcnt_var_eplained": float(ssd.pca.explained_variance_ratio_.sum() * 100),
    }
    df_stats = pd.DataFrame([stats])
    df_stats = df_stats[
        [
            "Trait",
            "PCA_K",
            "R2",
            "adj_R2",
            "F",
            "p",
            "beta_norm",
            "delta_per_0.10_raw",
            "IQR_effect_raw",
            "corr_y_pred",
            "prcnt_var_eplained",
        ]
    ]
    df_stats.columns = [
        "Trait",
        "PCA_K",
        "R²",
        "Adj. R²",
        "F-stat",
        "p-value",
        "‖β‖",
        "Δ per 0.1",
        "IQR",
        "r",
        "% Var Explained",
    ]

    # sanitize before saving
    df_stats = sanitize_df_for_excel(df_stats)

    df_stats.to_excel(
        os.path.join(out_dir, f"ssd_stats_summary_{prefix}.xlsx"),
        index=False,
    )

    # ---- scores
    scores = ssd.ssd_scores()
    scores["text"] = texts

    scores = sanitize_df_for_excel(scores)

    scores.to_excel(
        os.path.join(out_dir, f"ssd_scores_{prefix}.xlsx"),
        index=False,
    )

    # ---- clusters/members/snippets
    df_clusters, df_members = ssd.cluster_neighbors(**FULL_CLUSTER_PARAMS)

    df_members = sanitize_df_for_excel(df_members)

    df_members.to_excel(
        os.path.join(out_dir, f"cluster_members_{prefix}.xlsx"),
        index=False,
    )

    sn_clust = ssd.cluster_snippets(pre_docs=pre_docs, top_per_cluster=10, side="both")
    df_pos_snip = sn_clust["pos"]
    df_neg_snip = sn_clust["neg"]

    df_clusters = build_representative_snippets(df_clusters, df_pos_snip, df_neg_snip)

    keep_cols = [
        "side",
        "cluster_rank",
        "size",
        "centroid_cos_beta",
        "coherence",
        "top_words",
        "representative_snippet",
    ]
    keep_cols = [c for c in keep_cols if c in df_clusters.columns]
    df_clusters = df_clusters[keep_cols]

    expected = [
        "side",
        "cluster_rank",
        "size",
        "centroid_cos_beta",
        "coherence",
        "top_words",
        "representative_snippet",
    ]
    if all(c in df_clusters.columns for c in expected):
        df_clusters = df_clusters[expected]

    pos_clusters, neg_clusters = format_clusters_tables(df_clusters)

    pos_clusters = sanitize_df_for_excel(pos_clusters)
    neg_clusters = sanitize_df_for_excel(neg_clusters)

    pos_clusters.to_excel(
        os.path.join(out_dir, f"clusters_pos_{prefix}.xlsx"),
        index=False,
    )
    neg_clusters.to_excel(
        os.path.join(out_dir, f"clusters_neg_{prefix}.xlsx"),
        index=False,
    )

    save_snippets_by_cluster(df_pos_snip, out_dir, prefix)
    save_snippets_by_cluster(df_neg_snip, out_dir, prefix)


# -----------------------
# MAIN
# -----------------------
print(f"\n=== MODEL: {MODEL_NAME} ===")
model_dir = os.path.join(RESULTS_DIR, MODEL_NAME)
os.makedirs(model_dir, exist_ok=True)

sweep_root = os.path.join(model_dir, "sweep")
fixed120_root = os.path.join(model_dir, "120")
os.makedirs(sweep_root, exist_ok=True)
os.makedirs(fixed120_root, exist_ok=True)

print("Loading embeddings...")
kv = load_embeddings(MODEL_PATH)

import numpy as np
from gensim.models import KeyedVectors

norms = np.linalg.norm(kv.vectors, axis=1)
keep = norms > 1e-12

if not np.all(keep):
    kv2 = KeyedVectors(vector_size=kv.vector_size)
    kv2.add_vectors(
        [w for w, ok in zip(kv.index_to_key, keep) if ok],
        kv.vectors[keep],
    )
    kv2.fill_norms()
    kv = kv2

kv = normalize_kv(kv, l2=True, abtt_m=1)

print(f"\n--- SLICE: {SLICE_NAME} ({TEXT_COL}) ---")

need_cols = [TEXT_COL] + list(TRAITS.values())
dfi = df_all.dropna(subset=need_cols).reset_index(drop=True)
if len(dfi) == 0:
    raise ValueError(f"No rows after dropna for slice {SLICE_NAME}")

texts = dfi[TEXT_COL].tolist()

print("Preprocessing texts once for this slice...")
pre_docs = preprocess_texts(texts, nlp, stopwords, show_progress=True)
docs = build_docs_from_preprocessed(pre_docs)

for trait_name, y_col in TRAITS.items():
    print(f"\n   >>> TRAIT: {trait_name}")
    y = dfi[y_col].to_numpy()

    # Trait-specific directories:
    #   - sweep_root/<trait_name>  -> sweep + best-K SSD
    #   - fixed120_root/<trait_name> -> SSD with K=120
    trait_sweep_dir = os.path.join(sweep_root, trait_name)
    os.makedirs(trait_sweep_dir, exist_ok=True)

    trait_120_dir = os.path.join(fixed120_root, trait_name)
    os.makedirs(trait_120_dir, exist_ok=True)

    prefix = f"{MODEL_NAME}_{SLICE_NAME}_{trait_name}"

    # 1) Sweep + selection (no extra plotting beyond SWEEP_* flags)
    result = pca_sweep(
        kv=kv,
        docs=docs,
        y=y,
        lexicon=SSD_LEXICON,
        use_full_doc=SSD_USE_FULLDOC,
        pca_k_values=PCA_K_VALUES,
        sif_a=SSD_SIF_A,
        cluster_topn=SWEEP_CLUSTER_TOPN,
        k_min=SWEEP_K_MIN,
        k_max=SWEEP_K_MAX,
        top_words=SWEEP_TOP_WORDS,
        weight_by_size=SWEEP_WEIGHT_BY_SIZE,
        auck_radius=AUCK_RADIUS,
        beta_smooth_win=BETA_SMOOTH_WIN,
        beta_smooth_kind=BETA_SMOOTH_KIND,
        out_dir=trait_sweep_dir,
        prefix=prefix,
        save_tables=SAVE_SWEEP_TABLE,
        save_figures=SAVE_SWEEP_FIGURE,
        verbose=True,
    )
    best_k = result.best_k
    print(f"Selected PCA_K={best_k} (JOINT AUCK)")

    # 2) Final SSD run at chosen K -> save original SSD result tables to "sweep" folder
    run_and_save_final_ssd(
        kv=kv,
        docs=docs,
        pre_docs=pre_docs,
        texts=texts,
        y=y,
        trait_name=trait_name,
        pca_k=best_k,
        out_dir=trait_sweep_dir,
        prefix=prefix,
    )

    # 3) Additional SSD run with fixed PCA_K = 120 -> save to "120" folder
    prefix_120 = f"{prefix}_K120"
    run_and_save_final_ssd(
        kv=kv,
        docs=docs,
        pre_docs=pre_docs,
        texts=texts,
        y=y,
        trait_name=trait_name,
        pca_k=120,
        out_dir=trait_120_dir,
        prefix=prefix_120,
    )

print("\nAll done.")
