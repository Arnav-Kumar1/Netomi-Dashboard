import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from itertools import chain

# Set Streamlit page config
st.set_page_config(
    page_title="Customer Support KPI Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Customer Support KPI Dashboard")
st.markdown("Analyze trends, topics, and entities from customer query data")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("clustered_labelled_queries_final.csv")
    entity_df = pd.read_csv("identified_entities.csv")
    return df, entity_df

df, entity_df = load_data()

# Sidebar
st.sidebar.header("Filters")

# KPI Section
st.markdown("## 🔑 Key Performance Indicators")
st.markdown("Below are the most critical metrics helping us understand customer support trends:")

col1, col2, col3 = st.columns(3)
col1.metric("📦 Total Queries", len(df))
col2.metric("🧩 Unique Topics", df['topic_label_final'].nunique())

# True Queries With Entities (based on Query Rows union, excluding ignored types)
valid_entity_df = entity_df[~entity_df['Entity Type'].isin(['CARDINAL', 'ORG', 'TIME'])]
all_rows = valid_entity_df['Query Rows'].dropna().apply(ast.literal_eval)
all_indices = set(chain.from_iterable(all_rows))
num_queries_with_entities = len(all_indices)
col3.metric("🧾 Queries With Entities", num_queries_with_entities)

# Optional: Total Mentions as a fourth KPI
col4, _, _ = st.columns(3)
total_entity_mentions = valid_entity_df['Frequency'].sum()
col4.metric("🔢 Total Entity Mentions", total_entity_mentions)

st.markdown("---")

# Topic Counts Bar Chart
st.markdown("## 📈 Topic Frequency Distribution")
N = st.slider("Select number of top topics to display", min_value=5, max_value=len(df['topic_label_final'].unique()), value=10)

def show_topic_counts(df, N=10):
    topic_counts = df['topic_label_final'].value_counts().sort_values(ascending=False)
    top_N = topic_counts.head(N)

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(top_N.index, top_N.values, color="skyblue")

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval), ha='center', va='bottom')

    ax.set_title(f"Top {N} Topics by Frequency")
    ax.set_ylabel("Query Count")
    ax.set_xlabel("Topic")
    ax.set_xticklabels(top_N.index, rotation=45, ha="right")

    st.pyplot(fig)

    # Pareto chart
    st.markdown("## 📉 Pareto Analysis of Topic Frequency")
    cumulative_percentage = topic_counts.cumsum() / topic_counts.sum() * 100

    fig2, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(topic_counts.index, topic_counts.values, color="skyblue")
    ax2 = ax1.twinx()
    ax2.plot(topic_counts.index, cumulative_percentage, color="red", marker="o")
    ax2.axhline(y=80, color='gray', linestyle='--')
    ax2.text(len(topic_counts)*0.8, 82, '80% Line', color='gray')

    ax1.set_ylabel("Query Count")
    ax2.set_ylabel("Cumulative %")
    ax1.set_title("Pareto Chart: Topic Contribution to Total Volume")
    ax1.set_xticklabels(topic_counts.index, rotation=45, ha="right")
    st.pyplot(fig2)

show_topic_counts(df, N=N)

# Sample Queries by Topic
st.markdown("## 📝 Sample Queries by Topic")
sample_topic = st.selectbox("Select Topic for Sample Queries", sorted(df['topic_label_final'].unique()))

def show_sample_queries(df, topic, samples_per_topic=5):
    group = df[df['topic_label_final'] == topic]
    st.markdown(f"**{topic}**")
    examples = group['raw_query'].dropna().sample(min(len(group), samples_per_topic), random_state=42).tolist()
    for q in examples:
        st.markdown(f"- {q}")

show_sample_queries(df, sample_topic)

# Entity Type Distribution Plot (Excluding CARDINAL, ORG, TIME)
def plot_entity_distribution(entity_df):
    exclude_types = {"CARDINAL", "ORG", "TIME"}
    entity_df = entity_df[~entity_df["Entity Type"].isin(exclude_types)]

    # Aggregate total frequency for each entity type
    type_freq = entity_df.groupby("Entity Type")["Frequency"].sum().sort_values(ascending=False)

    # Plot
    st.markdown("## 🧠 Entity Type Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    type_freq.plot(kind="bar", color="lightgreen", ax=ax)

    for i, (etype, freq) in enumerate(type_freq.items()):
        ax.text(i, freq + 0.5, str(freq), ha='center', va='bottom', fontsize=9)

    ax.set_ylabel("Total Frequency")
    ax.set_xlabel("Entity Type")
    ax.set_title("Entity Type Distribution (Excluding CARDINAL, ORG, TIME)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

plot_entity_distribution(entity_df)
