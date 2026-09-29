import json
from data import PASSAGES, QUESTIONS

texts = {pid: text for pid, text in PASSAGES}

def tokset(s):
    import re
    return set(re.findall(r"[a-z0-9]+", s.lower())) - {
        "the","a","an","is","was","were","of","in","on","to","and","by","for","which",
        "who","what","when","where","its","it","as","at","that","with","his","her"
    }

rows = []
for pid, q, qtype in QUESTIONS:
    ptoks = tokset(texts[pid])
    qtoks = tokset(q)
    overlap = qtoks & ptoks
    rows.append((pid, qtype, len(qtoks), len(overlap), round(len(overlap)/max(len(qtoks),1), 3)))

import pandas as pd
df = pd.DataFrame(rows, columns=["passage_id","type","q_content_words","shared_words","overlap_ratio"])
print(df.groupby("type")["overlap_ratio"].mean())
df.to_csv("overlap.csv", index=False)
