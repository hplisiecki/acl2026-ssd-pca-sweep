import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from matplotlib.patches import FancyArrowPatch
from matplotlib.legend_handler import HandlerPatch

from ssdiff import (
    SSD,
    load_embeddings,
    normalize_kv,
    load_spacy,
    load_stopwords,
    preprocess_texts,
    build_docs_from_preprocessed,
)

SAVE_COMPARISON_FIGURE = True

from pca_sweep import pca_sweep

# -----------------------
# CONFIG
# -----------------------
DF_PATH = r"data\AI_POSTS.xlsx"
RESULTS_DIR = r"results"

# Use only Dolma embeddings
MODEL_NAME = "dolma_2024"
MODEL_PATH = r"D:\Resources\NLP\EN\glove.2024.dolma.300d\dolma_300_2024_1.2M.100_combined.kv"

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

class HandlerArrow(HandlerPatch):
    def create_artists(
        self,
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
    ):
        p = FancyArrowPatch(
            (xdescent, ydescent + height / 2),
            (xdescent + width, ydescent + height / 2),
            arrowstyle='->',
            mutation_scale=12,
            lw=1.5,
        )
        p.set_transform(trans)
        return [p]

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

# -----------------------
# HELPERS (cluster visualization)
# -----------------------

def _find_first_existing(df: pd.DataFrame, candidates: list[str], required: bool = True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(
            f"None of the expected columns found. Tried: {candidates}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def _infer_members_columns(df_members: pd.DataFrame) -> dict:
    """
    Try to infer standard column names from df_members returned by ssd.cluster_neighbors().
    Adjust candidate lists here if your package uses different names.
    """
    word_col = _find_first_existing(
        df_members,
        ["word", "term", "token", "neighbor", "item", "label"]
    )
    side_col = _find_first_existing(
        df_members,
        ["side", "pole", "sign"]
    )
    cluster_col = _find_first_existing(
        df_members,
        ["cluster_rank", "cluster", "cluster_id", "cluster_no", "rank"]
    )
    cosine_col = _find_first_existing(
        df_members,
        ["cosine", "cos_beta", "cos_to_beta", "cos_sim", "similarity"],
        required=False,
    )

    return {
        "word": word_col,
        "side": side_col,
        "cluster": cluster_col,
        "cosine": cosine_col,
    }


def _standardize_members_df(df_members: pd.DataFrame) -> pd.DataFrame:
    """
    Return a standardized plotting dataframe with columns:
      word, side, cluster_rank, cosine
    """
    cols = _infer_members_columns(df_members)
    out = df_members.copy()

    out = out.rename(
        columns={
            cols["word"]: "word",
            cols["side"]: "side",
            cols["cluster"]: "cluster_rank",
        }
    )

    if cols["cosine"] is not None:
        out = out.rename(columns={cols["cosine"]: "cosine"})
    else:
        out["cosine"] = np.nan

    out["word"] = out["word"].astype(str)
    out["side"] = out["side"].astype(str).str.lower().str.strip()

    # normalize common aliases just in case
    out["side"] = out["side"].replace({
        "positive": "pos",
        "+": "pos",
        "negative": "neg",
        "-": "neg",
    })

    out["cluster_rank"] = pd.to_numeric(out["cluster_rank"], errors="coerce").astype("Int64")

    keep = ["word", "side", "cluster_rank", "cosine"]
    keep = [c for c in keep if c in out.columns]
    return out[keep].drop_duplicates().reset_index(drop=True)


def _build_plot_df_from_members(kv, df_members: pd.DataFrame) -> pd.DataFrame:
    """
    Build a plotting dataframe by attaching word vectors from kv to df_members.
    """
    dm = _standardize_members_df(df_members)

    rows = []
    missing = []

    for _, r in dm.iterrows():
        w = r["word"]
        if w in kv.key_to_index:
            vec = kv[w]
            rows.append({
                "word": w,
                "side": r["side"],
                "cluster_rank": int(r["cluster_rank"]) if pd.notnull(r["cluster_rank"]) else -1,
                "cosine": float(r["cosine"]) if pd.notnull(r["cosine"]) else np.nan,
                "vec": vec,
            })
        else:
            missing.append(w)

    plot_df = pd.DataFrame(rows)

    if len(plot_df) == 0:
        raise ValueError("No member words were found in kv. Cannot build visualization.")

    if missing:
        print(f"[viz] Warning: {len(missing)} member words missing from kv. "
              f"Example missing: {missing[:10]}")

    # local PCA projection on the retrieved words only
    X = np.vstack(plot_df["vec"].to_numpy())
    pca2 = PCA(n_components=2, random_state=0)
    XY = pca2.fit_transform(X)

    plot_df["x2"] = XY[:, 0]
    plot_df["y2"] = XY[:, 1]

    # useful combined label
    plot_df["cluster_label"] = plot_df.apply(
        lambda r: f"{r['side']}_cluster_{int(r['cluster_rank'])}",
        axis=1,
    )

    return plot_df


def _project_beta_to_local_2d(kv, beta_vec: np.ndarray, plot_df: pd.DataFrame) -> np.ndarray:
    """
    Project the original-space beta vector into the same local PCA-2D space
    used for the retrieved member words.
    """
    X = np.vstack(plot_df["vec"].to_numpy())
    pca2 = PCA(n_components=2, random_state=0)
    pca2.fit(X)

    beta_vec = np.asarray(beta_vec, dtype=float)
    beta_norm = np.linalg.norm(beta_vec)
    if beta_norm == 0 or not np.isfinite(beta_norm):
        return np.array([0.0, 0.0])

    beta_unit = beta_vec / beta_norm
    beta_2d = pca2.components_ @ beta_unit  # shape (2,)

    norm2 = np.linalg.norm(beta_2d)
    if norm2 == 0 or not np.isfinite(norm2):
        return np.array([0.0, 0.0])

    return beta_2d / norm2


def _pick_words_to_label(plot_df: pd.DataFrame, top_per_cluster: int = 4, max_total: int = 36) -> pd.DataFrame:
    """
    Pick a manageable subset of words for labeling.
    Preference:
      - highest cosine if available
      - otherwise points farthest from origin
    """
    d = plot_df.copy()

    if d["cosine"].notna().any():
        d["_priority"] = d["cosine"].fillna(-np.inf)
    else:
        d["_priority"] = np.sqrt(d["x2"] ** 2 + d["y2"] ** 2)

    label_rows = []
    for _, sub in d.groupby(["side", "cluster_rank"], dropna=False):
        sub = sub.sort_values("_priority", ascending=False).head(top_per_cluster)
        label_rows.append(sub)

    if len(label_rows) == 0:
        return d.head(0)

    lab = pd.concat(label_rows, ignore_index=True)
    lab = lab.sort_values("_priority", ascending=False).head(max_total)
    return lab


def _draw_cluster_panel(
    ax,
    plot_df: pd.DataFrame,
    beta_2d: np.ndarray | None,
    title: str,
    subtitle: str | None = None,
):
    """
    Draw one panel of the comparison figure.
    Uses cluster color and side-specific marker.
    """
    # stable cluster ordering for consistent colors
    unique_clusters = list(plot_df["cluster_label"].drop_duplicates())
    cmap = plt.get_cmap("tab10")
    color_map = {cl: cmap(i % 10) for i, cl in enumerate(unique_clusters)}

    marker_map = {"pos": "o", "neg": "s"}

    # scatter by cluster
    for cl, sub in plot_df.groupby("cluster_label", dropna=False):
        side = sub["side"].iloc[0]
        ax.scatter(
            sub["x2"],
            sub["y2"],
            s=45,
            alpha=0.75,
            c=[color_map[cl]],
            marker=marker_map.get(side, "o"),
            edgecolors="black",
            linewidths=0.3,
        )



    ax.axhline(0, linewidth=0.8, alpha=0.3)
    ax.axvline(0, linewidth=0.8, alpha=0.3)

    ttl = title if subtitle is None else f"{title}\n{subtitle}"
    ax.set_title(ttl, fontsize=11)
    ax.set_xlabel("Local PCA-1")
    ax.set_ylabel("Local PCA-2")

    # make aspect visually pleasant
    ax.margins(0.08)


def save_cluster_comparison_figure(
    *,
    kv,
    ssd_best,
    df_members_best: pd.DataFrame,
    best_k: int,
    ssd_high,
    df_members_high: pd.DataFrame,
    high_k: int,
    out_path: str,
    trait_name: str,
    model_name: str,
    slice_name: str,
):
    """
    Save a 2-panel figure comparing retrieved cluster neighborhoods
    for the sweep-selected solution and the high-K counterfactual.
    """
    plot_best = _build_plot_df_from_members(kv, df_members_best)
    plot_high = _build_plot_df_from_members(kv, df_members_high)

    # try to get original-space beta vector from SSD object
    beta_best = None
    beta_high = None

    for attr in ["beta", "beta_raw", "beta_backprojected", "beta_vec", "beta_original_space"]:
        if beta_best is None and hasattr(ssd_best, attr):
            beta_best = getattr(ssd_best, attr)
        if beta_high is None and hasattr(ssd_high, attr):
            beta_high = getattr(ssd_high, attr)

    beta2_best = _project_beta_to_local_2d(kv, beta_best, plot_best) if beta_best is not None else None
    beta2_high = _project_beta_to_local_2d(kv, beta_high, plot_high) if beta_high is not None else None

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

    _draw_cluster_panel(
        axes[0],
        plot_best,
        beta2_best,
        title=f"{trait_name}: sweep-selected solution",
        subtitle=f"K = {best_k}",
    )
    _draw_cluster_panel(
        axes[1],
        plot_high,
        beta2_high,
        title=f"{trait_name}: high-dimensional counterfactual",
        subtitle=f"K = {high_k}",
    )

    # side legend
    side_handles = [
        Line2D([0], [0], marker='o', linestyle='None', markeredgecolor='black',
               label='Positive pole', markersize=8),
        Line2D([0], [0], marker='s', linestyle='None', markeredgecolor='black',
               label='Negative pole', markersize=8),
    ]
    fig.legend(handles=side_handles, loc="lower center", ncol=2, frameon=False)

    fig.suptitle(
        f"SSD cluster comparison: {trait_name} | {model_name} | {slice_name}",
        fontsize=13
    )

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[viz] Saved comparison figure to: {out_path}")


def run_and_save_final_ssd(
    *,
    kv,
    docs,
    pre_docs,
    texts,
    y,
    trait_name: str,
    pca_k: int,
    out_dir: str,
    prefix: str,
    return_objects: bool = False,
):
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

    df_members_for_plot = df_members.copy()
    df_members_excel = sanitize_df_for_excel(df_members)

    df_members_excel.to_excel(
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

    if return_objects:
        return {
            "ssd": ssd,
            "df_clusters": df_clusters,
            "df_members": df_members_for_plot,
            "df_pos_snip": df_pos_snip,
            "df_neg_snip": df_neg_snip,
        }

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
    best_run = run_and_save_final_ssd(
        kv=kv,
        docs=docs,
        pre_docs=pre_docs,
        texts=texts,
        y=y,
        trait_name=trait_name,
        pca_k=best_k,
        out_dir=trait_sweep_dir,
        prefix=prefix,
        return_objects=True,
    )

    # 3) Additional SSD run with fixed PCA_K = 120 -> save to "120" folder
    prefix_120 = f"{prefix}_K120"
    high_run = run_and_save_final_ssd(
        kv=kv,
        docs=docs,
        pre_docs=pre_docs,
        texts=texts,
        y=y,
        trait_name=trait_name,
        pca_k=120,
        out_dir=trait_120_dir,
        prefix=prefix_120,
        return_objects=True,
    )

    # 4) Comparison figure: sweep-selected K vs K=120
    if SAVE_COMPARISON_FIGURE:
        comparison_path = os.path.join(
            model_dir,
            f"cluster_comparison_{prefix}.png"
        )

        save_cluster_comparison_figure(
            kv=kv,
            ssd_best=best_run["ssd"],
            df_members_best=best_run["df_members"],
            best_k=best_k,
            ssd_high=high_run["ssd"],
            df_members_high=high_run["df_members"],
            high_k=120,
            out_path=comparison_path,
            trait_name=trait_name,
            model_name=MODEL_NAME,
            slice_name=SLICE_NAME,
        )

print("\nAll done.")
