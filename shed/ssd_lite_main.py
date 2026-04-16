dir = r"D:\Github\Supervised-Semantic-Differential"

import sys
# append path
sys.path.append(dir)
import pandas as pd
from ssdiff import Embeddings, Corpus, SSD

MODEL_PATH = r"D:\Resources\NLP\EN\glove.2024.dolma.300d\dolma_300_2024_1.2M.100_combined.kv"


emb = Embeddings.load(MODEL_PATH)


emb.normalize(l2 = True, abtt_m = 1)

SLICE_NAME = "ai"          # kept only for printing
TEXT_COL = "text_ai"

TRAITS = {
    "ADM": "ADM",
    "RIV": "RIV",
}

DF_PATH = r"../data/AI_POSTS.xlsx"

df_all = pd.read_excel(DF_PATH)

# drop nans in TEXT_COL and TRAITS
dfi = df_all.dropna(subset=[TEXT_COL] + list(TRAITS.values())).reset_index(drop=True)

texts = dfi[TEXT_COL].tolist()

scores = dfi[TRAITS["ADM"]].values

corpus = Corpus(texts, lang="en")

from ssdiff.utils.lexicon import token_presence_stats, coverage_by_lexicon, suggest_lexicon

lexicon_suggest = corpus.suggest_lexicon(scores)
lexicon_suggest = suggest_lexicon((texts, scores))
lexicon = ['ai', 'think']
lexicon_coverage = coverage_by_lexicon((texts, scores), ['ai'], verbose = True)


ssd = SSD(emb, corpus, scores, lexicon)
print(ssd)


# PLS
pls = ssd.fit_pls()
print(pls)
print()
print(pls.summary())
print()
pls.report(k_min=2, k_max=5, top_words=10, clusters=50, extreme_docs=5, misdiagnosed=5)
print()
print(pls.split_test(n_splits=30))

clusters = pd.DataFrame(pls.cluster_neighbors(topn=100, k_min=2, k_max=2))
pls.snippets(corpus.pre_docs)


snippets = pls.cluster_snippets(corpus.pre_docs)


pls.cluster

# ── Group (median split) ────────────────────────────────────────
gr = ssd.fit_groups(median_split=True)
print(gr)
print()
print(gr.summary())
print()
gr.report(top_words=10, clusters=50)

# ── PCA+OLS sweep ───────────────────────────────────────────────
ols = ssd.fit_ols(k_min=2, k_max=120)
print(ols)
print()
print(ols.summary())


clusters_pos = pd.DataFrame(ols.cluster_neighbors(topn=100, k_min=2, k_max=10, side="pos"))
clusters_neg = pd.DataFrame(ols.cluster_neighbors(topn=100, k_min=2, k_max=10, side="neg"))

snippets = ols.cluster_snippets(pre_docs=corpus.pre_docs, recompute = True)

centroid_labels = [s['centroid_label'] for s in snippets['neg']]



ols.plot_sweep(path="sweep_plot.png")




print("Preprocessing texts once for this slice...")
pre_docs = preprocess_texts(texts, nlp, stopwords, show_progress=True)
docs = build_docs_from_preprocessed(pre_docs)