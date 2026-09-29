import json, re
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from data import PASSAGES, QUESTIONS

ids = [p[0] for p in PASSAGES]
texts = [p[1] for p in PASSAGES]
id2idx = {pid: i for i, pid in enumerate(ids)}

def tokenize(s):
    return re.findall(r"[a-z0-9]+", s.lower())

# ---------- BM25 (sparse / lexical) ----------
bm25 = BM25Okapi([tokenize(t) for t in texts])

def bm25_rank(question):
    scores = bm25.get_scores(tokenize(question))
    return np.argsort(-scores)

# ---------- TF-IDF + LSA (dense/semantic proxy) ----------
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(texts)
n_components = 15
svd = TruncatedSVD(n_components=n_components, random_state=42)
passage_vecs = svd.fit_transform(tfidf_matrix)

def lsa_rank(question):
    q_tfidf = vectorizer.transform([question])
    q_vec = svd.transform(q_tfidf)
    sims = cosine_similarity(q_vec, passage_vecs)[0]
    return np.argsort(-sims)

# ---------- Evaluation ----------
def evaluate(rank_fn):
    rows = []
    for pid, question, qtype in QUESTIONS:
        gold_idx = id2idx[pid]
        ranking = rank_fn(question)
        rank_pos = int(np.where(ranking == gold_idx)[0][0]) + 1  # 1-indexed
        rows.append({"passage_id": pid, "type": qtype, "rank": rank_pos})
    return pd.DataFrame(rows)

bm25_df = evaluate(bm25_rank)
lsa_df = evaluate(lsa_rank)

def summarize(df, method):
    out = {}
    for scope, sub in [("overall", df), ("matched", df[df.type == "matched"]), ("paraphrased", df[df.type == "paraphrased"])]:
        n = len(sub)
        out[scope] = {
            "n": n,
            "recall@1": round((sub["rank"] <= 1).mean(), 3),
            "recall@3": round((sub["rank"] <= 3).mean(), 3),
            "recall@5": round((sub["rank"] <= 5).mean(), 3),
            "mrr": round((1.0 / sub["rank"]).mean(), 3),
        }
    return {"method": method, **out}

results = {
    "bm25": summarize(bm25_df, "BM25 (lexical)"),
    "lsa": summarize(lsa_df, "TF-IDF+LSA (semantic proxy)"),
}

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

bm25_df.to_csv("bm25_ranks.csv", index=False)
lsa_df.to_csv("lsa_ranks.csv", index=False)

print(json.dumps(results, indent=2))
