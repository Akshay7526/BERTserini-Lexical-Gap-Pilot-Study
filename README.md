# BERTserini Lexical-Gap Pilot Study

Small-scale pilot experiment comparing BM25 (sparse/lexical) vs. TF-IDF+LSA (classical dense/semantic
proxy) retrieval on a hand-curated 24-passage / 48-question corpus, split into "matched" and
"paraphrased" question types, to probe whether BERTserini's independently-trained BM25 retriever is a
bottleneck on lexically divergent questions.

## Files
- `data.py` — the 24 passages and 48 questions (matched/paraphrased pairs)
- `run_experiment.py` — runs BM25 and TF-IDF+LSA retrieval, computes Recall@1/3/5 and MRR, writes `results.json`
- `overlap_analysis.py` — computes lexical overlap ratio between questions and gold passages
- `make_chart.py` — generates the results bar chart used on the poster

## Run
```bash
pip install rank_bm25 scikit-learn matplotlib pandas
python3 run_experiment.py
python3 overlap_analysis.py
python3 make_chart.py
```

## Note on scope
This pilot was built entirely offline (no pretrained neural retriever / no external dataset download)
due to environment constraints. TF-IDF+LSA is used as a transparent, reproducible proxy for a neural
dense retriever (e.g., DPR). See the poster's Limitations section for full discussion.
