import os
import re
import math
import zipfile
from collections import defaultdict, Counter
import pandas as pd
from nltk.stem import PorterStemmer

# ===============================
# PATH CONFIG
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CRANFIELD_DIR = os.path.join(BASE_DIR, "data", "Cranfield")
TEST_CSV = os.path.join(BASE_DIR, "data", "test.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "submission")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "submission.csv")
OUTPUT_ZIP = os.path.join(OUTPUT_DIR, "submission.zip")

TOP_K = 50

# BM25 parameters (good for Cranfield)
k1 = 1.8
b = 0.7

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

# ===============================
# STOPWORDS
# ===============================
STOPWORDS = {
    "the","is","at","which","on","and","a","an",
    "of","to","in","for","with","by","from",
    "that","this","are","was","be","as","it"
}

stemmer = PorterStemmer()

# ===============================
# TOKENIZE
# ===============================
def tokenize(text: str):

    tokens = TOKEN_PATTERN.findall(text.lower())

    tokens = [
        stemmer.stem(t)
        for t in tokens
        if t not in STOPWORDS
    ]

    return tokens


# ===============================
# LOAD DOCUMENTS
# ===============================
def load_documents(folder_path):

    documents = {}

    for filename in os.listdir(folder_path):

        if filename.endswith(".txt"):

            doc_id = int(os.path.splitext(filename)[0])
            path = os.path.join(folder_path, filename)

            with open(
                path,
                "r",
                encoding="utf-8",
                errors="ignore"
            ) as f:
                documents[doc_id] = f.read()

    print(f"Loaded {len(documents)} documents")
    return documents


# ===============================
# BUILD BM25 INDEX
# ===============================
def build_bm25_index(documents):

    inverted_index = defaultdict(dict)
    doc_len = {}
    df = defaultdict(int)

    for doc_id, text in documents.items():

        terms = tokenize(text)
        doc_len[doc_id] = len(terms)

        tf = Counter(terms)

        for term, freq in tf.items():
            inverted_index[term][doc_id] = freq
            df[term] += 1

    N = len(documents)
    avgdl = sum(doc_len.values()) / N

    idf = {}

    for term, freq in df.items():
        idf[term] = math.log(
            (N - freq + 0.5) / (freq + 0.5) + 1
        )

    print("BM25 index built")
    return inverted_index, doc_len, avgdl, idf


# ===============================
# BM25 RETRIEVE (CORRECT VERSION)
# ===============================
def retrieve(query,
             inverted_index,
             doc_len,
             avgdl,
             idf):

    scores = defaultdict(float)

    q_terms = tokenize(query)
    q_freq = Counter(q_terms)

    for term, qf in q_freq.items():

        if term not in inverted_index:
            continue

        for doc_id, tf in inverted_index[term].items():

            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (
                1 - b + b * doc_len[doc_id] / avgdl
            )

            score = idf[term] * (numerator / denominator)

            # query frequency weight
            scores[doc_id] += qf * score

    ranked_docs = sorted(
        scores.items(),
        key=lambda x: (-x[1], x[0])
    )

    return [doc for doc, _ in ranked_docs[:TOP_K]]


# ===============================
# MAIN
# ===============================
def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    documents = load_documents(CRANFIELD_DIR)

    inverted_index, doc_len, avgdl, idf = \
        build_bm25_index(documents)

    queries = pd.read_csv(TEST_CSV)

    results = []

    for _, row in queries.iterrows():

        qid = int(row["query_id"])
        qtext = row["query"]

        docs = retrieve(
            qtext,
            inverted_index,
            doc_len,
            avgdl,
            idf
        )

        results.append({
            "query_id": qid,
            "query": qtext,
            "relevant_docs": " ".join(map(str, docs))
        })

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)

    with zipfile.ZipFile(
        OUTPUT_ZIP,
        "w",
        zipfile.ZIP_DEFLATED
    ) as zf:
        zf.write(OUTPUT_CSV, "submission.csv")

    print("✅ Done!")
    print(f"Saved to: {OUTPUT_ZIP}")


if __name__ == "__main__":
    main()