# Bertserini-lexical-gap-pilot

Quick pilot study I ran for my NLP poster on BERTserini (Yang et al. 2019). the paper pairs BM25 with a
BERT reader trained totally separately from each other, and I wanted to know if that's actually the
bottleneck people say it is - specifically whether BM25 falls apart when a question is phrased
differently from the passage that answers it (the "lexical gap" thing).

Original plan was to test BM25 vs an actual dense retriever (DPR or similar) but I didn't have internet
access to pull pretrained models in my dev environment and the deadline was 3 days out, so I used
TF-IDF + LSA instead as a stand-in for "semantic" retrieval. it's not a neural model but it's still a
dense, compressed representation and it's fully offline/reproducible, so it felt like a reasonable
compromise given the constraints. this is flagged clearly on the poster too, not hiding it.

## What's actually in here

- `data.py` - 24 passages I wrote by hand (mix of history/science/geography/etc so they don't overlap
  topically) + 48 questions, 2 per passage. one question per pair reuses words from the passage
  ("matched"), the other is a paraphrase that avoids the same wording where possible ("paraphrased").
- `run_experiment.py` - the actual experiment. runs both retrieval methods over all 48 questions, checks
  where the correct passage landed in the ranking, spits out recall@1/3/5 + MRR into results.json
- `overlap_analysis.py` - side analysis, just checks how much word overlap the paraphrased questions
  actually ended up having with their passage (spoiler: more than I wanted, see below)
- `make_chart.py` - makes the bar chart that's on the poster

## How to run it?

```
pip install rank_bm25 scikit-learn matplotlib pandas
python3 run_experiment.py
python3 overlap_analysis.py
python3 make_chart.py
```

No other setup needed, everything runs offline once the packages are installed.

## tl;dr of the results

Honestly not what I expected going in. BM25 held up fine even on the paraphrased questions
(recall@1 = 0.96 vs 0.92 for TF-IDF+LSA) - basically the opposite of the hypothesis. checked the
overlap numbers afterward and even my "paraphrased" questions still shared a decent chunk of vocabulary
with the source passage (avg overlap ratio ~0.38 vs ~0.68 for the matched ones - lower, but not zero).
turns out writing questions with literally zero shared words is harder than it sounds, especially when
the passage has specific named entities in it. also 24 passages on completely different topics is a
pretty easy retrieval task for anything, lexical or not, so this probably isn't the corpus where the
"lexical gap" problem shows up.

Not claiming this disproves anything about BERTserini or DPR at scale - it's a 24-passage pilot, not a
benchmark. full discussion + limitations are on the poster (results/discussion panel), I just didn't
want to duplicate all of that here.

## If I had more time -

would've liked to run this against a real dense encoder + a bigger, messier corpus where passages
actually overlap in topic (that's probably where BM25 really starts to lose). might revisit this later,
not sure yet.
