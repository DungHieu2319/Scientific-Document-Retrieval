import os
import re
import zipfile
from collections import defaultdict
import pandas as pd

CRANFIELD_ZIP = "Cranfield.zip"
TEST_CSV = "test.csv"
OUTPUT_CSV = "submission.csv"
OUTPUT_ZIP = "submission.zip"
TOP_K = 50

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

def tokenize(text: str):
    return TOKEN_PATTERN.findall(text.lower())


def load_documents(zip_path):
    documents = {}
    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            name = name.replace("\\", "/")
            if name.startswith("Cranfield/") and name.endswith(".txt"):
                doc_id = int(os.path.splitext(os.path.basename(name))[0])
                with z.open(name) as f:
                    documents[doc_id] = f.read().decode("utf-8", errors="ignore")
    print(f"Loaded {len(documents)} documents")
    return documents


def build_inverted_index(documents):
    inverted_index = defaultdict(set)
    doc_terms = {}

    for doc_id, text in documents.items():
        terms = tokenize(text)
        doc_terms[doc_id] = set(terms)
        for term in terms:
            inverted_index[term].add(doc_id)

    print(f"Inverted index size: {len(inverted_index)} terms")
    return inverted_index, doc_terms


def retrieve(query, inverted_index, doc_terms):
    q_terms = tokenize(query)

    candidate_docs = set()
    for term in q_terms:
        candidate_docs |= inverted_index.get(term, set())

    scored = []
    for doc_id in candidate_docs:
        score = sum(1 for t in q_terms if t in doc_terms[doc_id])
        scored.append((score, doc_id))

    scored.sort(key=lambda x: (-x[0], x[1]))

    return [doc_id for _, doc_id in scored[:TOP_K]]


def main():
    documents = load_documents(CRANFIELD_ZIP)
    inverted_index, doc_terms = build_inverted_index(documents)

    queries = pd.read_csv(TEST_CSV)
    results = []

    for _, row in queries.iterrows():
        qid = int(row["query_id"])
        qtext = row["query"]

        relevant_docs = retrieve(qtext, inverted_index, doc_terms)
        relevant_docs_str = " ".join(str(d) for d in relevant_docs)

        results.append({
            "query_id": qid,
            "query": qtext,
            "relevant_docs": relevant_docs_str
        })

    df_out = pd.DataFrame(results)
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(OUTPUT_CSV, OUTPUT_CSV)

    print(f"Done! Submission saved to {OUTPUT_ZIP}")


if __name__ == "__main__":
    main()
