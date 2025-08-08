import pandas as pd
from collections import Counter

# Load your labelled data and entity file
df = pd.read_csv("clustered_labelled_queries_final.csv")
entity_df = pd.read_csv("identified_entities.csv")

# Extract sets of queries with entities from entity_df
import ast
all_entity_rows = set()
for rows_str in entity_df['Query Rows'].dropna():
    rows = ast.literal_eval(rows_str)
    all_entity_rows.update([r - 2 for r in rows])  # adjust indexing as per your data

# Find queries with no entities based on df index
no_entity_df = df[~df.index.isin(all_entity_rows)]
no_entity_queries = no_entity_df['cleaned_query'].dropna().tolist()

def extract_ngrams(queries, n=2):
    """Extract n-grams from list of queries."""
    ngram_counter = Counter()
    for q in queries:
        words = q.lower().split()
        for i in range(len(words)-n+1):
            ngram = " ".join(words[i:i+n])
            ngram_counter[ngram] += 1
    return ngram_counter

# Get common single words and bigrams in no-entity queries
single_words = extract_ngrams(no_entity_queries, n=1)
bigrams = extract_ngrams(no_entity_queries, n=2)

print("Top 30 single words in no-entity queries:")
for w, c in single_words.most_common(30):
    print(f"{w}: {c}")

print("\nTop 30 two-word phrases in no-entity queries:")
for p, c in bigrams.most_common(30):
    print(f"{p}: {c}")
