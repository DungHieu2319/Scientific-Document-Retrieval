import os
print(os.getcwd())
import re
import math
from collections import defaultdict, Counter
import pandas as pd
import zipfile

# ===============================
# PATH CONFIG
# ===============================
CRANFIELD_DIR = "data/Cranfield"
TEST_CSV = "data/test.csv"
OUTPUT_CSV = "submission/submission.csv"
OUTPUT_ZIP = "submission/submission.zip"

TOP_K = 50

# BM25 parameters
k1 = 1.5
b = 0.75

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


# ===============================
# TOKENIZE
# ===============================
def tokenize(text: str):
    return TOKEN_PATTERN.findall(text.lower())


# ===============================
# LOAD DOCUMENTS FROM FOLDER
# ===============================
def load_documents(folder_path):

    documents = {}

    for filename in os.listdir(folder_path):

        if filename.endswith(".txt"):

            # nếu tên file là 1.txt, 2.txt ...
            doc_id = int(os.path.splitext(filename)[0])

            file_path = os.path.join(folder_path, filename)

            with open(
                file_path,
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

        term_freq = Counter(terms)

        for term, freq in term_freq.items():
            inverted_index[term][doc_id] = freq
            df[term] += 1

    N = len(documents)
    avgdl = sum(doc_len.values()) / N

    # compute IDF
    idf = {}
    for term, freq in df.items():
        idf[term] = math.log(
            (N - freq + 0.5) / (freq + 0.5) + 1
        )

    print("BM25 index built")
    return inverted_index, doc_len, avgdl, idf


# ===============================
# BM25 RETRIEVE
# ===============================
def retrieve(query,
             inverted_index,
             doc_len,
             avgdl,
             idf):

    q_terms = tokenize(query)
    scores = defaultdict(float)

    for term in q_terms:

        if term not in inverted_index:
            continue

        for doc_id, tf in inverted_index[term].items():

            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (
                1 - b + b * doc_len[doc_id] / avgdl
            )

            score = idf[term] * numerator / denominator
            scores[doc_id] += score

    ranked_docs = sorted(
        scores.items(),
        key=lambda x: (-x[1], x[0])
    )

    return [doc_id for doc_id, _ in ranked_docs[:TOP_K]]


# ===============================
# MAIN PIPELINE
# ===============================
def main():

    # tạo folder submission nếu chưa có
    os.makedirs("submission", exist_ok=True)

    # load documents
    documents = load_documents(CRANFIELD_DIR)

    # build bm25
    inverted_index, doc_len, avgdl, idf = \
        build_bm25_index(documents)

    # load queries
    queries = pd.read_csv(TEST_CSV)

    results = []

    for _, row in queries.iterrows():

        qid = int(row["query_id"])
        qtext = row["query"]

        relevant_docs = retrieve(
            qtext,
            inverted_index,
            doc_len,
            avgdl,
            idf
        )

        results.append({
            "query_id": qid,
            "query": qtext,
            "relevant_docs":
                " ".join(map(str, relevant_docs))
        })

    df_out = pd.DataFrame(results)

    df_out.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8"
    )

    # zip submission
    with zipfile.ZipFile(
        OUTPUT_ZIP,
        "w",
        zipfile.ZIP_DEFLATED
    ) as zf:
        zf.write(OUTPUT_CSV, "submission.csv")

    print("✅ Done!")
    print(f"Saved: {OUTPUT_ZIP}")


if __name__ == "__main__":
    main()