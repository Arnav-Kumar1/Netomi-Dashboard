# step_2_topic_discovery.ipynb (consultant-grade modular pipeline)

import pandas as pd

from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# -------------------- 1. Load Cleaned Data --------------------
def load_cleaned_queries(filepath):
    df = pd.read_csv(filepath)
    if 'cleaned_query' not in df.columns:
        raise ValueError("Missing 'cleaned_query' column in uploaded file.")
    return df
# -------------------- 2. Generate Sentence Embeddings --------------------
def embed_queries(queries):
    # model = SentenceTransformer("all-MiniLM-L6-v2")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    embeddings = model.encode(queries.tolist(), show_progress_bar=True)
    return embeddings


def cluster_queries_dbscan(embeddings, eps=0.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = dbscan.fit_predict(embeddings)
    return labels


# -------------------- 4. Add Topics --------------------
def assign_topics(df, labels):
    df['topic'] = labels
    return df

# -------------------- 7. Save Output --------------------
def save_labeled_queries(df, output_path="clustered_queries_auto.csv"):
    df.to_csv(output_path, index=False)
    print(f"\n✅ Clustered data saved to: {output_path}")



# -------------------- 8. Run Full Step 2 Pipeline --------------------
def run_step_2_pipeline():
    df = load_cleaned_queries("cleaned_queries.csv")
    embeddings = embed_queries(df['cleaned_query'])
    db = DBSCAN(eps=0.5, min_samples=5, metric='euclidean')
    labels = db.fit_predict(embeddings)
    df = assign_topics(df, labels)
    save_labeled_queries(df)


run_step_2_pipeline()