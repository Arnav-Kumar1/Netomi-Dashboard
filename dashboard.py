import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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

# Add explanatory section on Topics vs Entities
st.markdown("""
### Understanding Topics & Entities

- **Topics** represent the broad intention or category of a customer's query, what they want to do (e.g., request refund, check order status).
- **Entities** are the key pieces of information within those queries, the details needed to fulfill the request (e.g., refund amount, order date, payment method).

This dashboard helps monitor both the variety of customer intents and the richness of their detailed information.
""")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("clustered_labelled_queries_final.csv")
    entity_df = pd.read_csv("identified_entities.csv")
    return df, entity_df

df, entity_df = load_data()

# Sidebar
st.sidebar.header("Filters")
# Placeholder for filters (date range, topics, entities, etc.) if needed in future


# KPI Section
st.markdown("## 🔑 Key Performance Indicators")
st.markdown("Below are the most critical metrics helping us understand customer support trends:")

col1, col2, col3 = st.columns(3)
col1.metric("📦 Total Queries", len(df))
col2.metric("🧩 Unique Topics", df['topic_label_final'].nunique())

# True Queries With Entities (based on Query Rows union, excluding ignored entity types)
valid_entity_df = entity_df[~entity_df['Entity Type'].isin(['CARDINAL', 'ORG', 'TIME'])]
all_rows = valid_entity_df['Query Rows'].dropna().apply(ast.literal_eval)
all_indices = set(chain.from_iterable(all_rows))
num_queries_with_entities = len(all_indices)
col3.metric("🧾 Queries With Entities", num_queries_with_entities)

# Optional: Total Mentions as a fourth KPI in new row to avoid crowding
st.markdown("")
col4, _, _ = st.columns(3)
total_entity_mentions = valid_entity_df['Frequency'].sum()
col4.metric("🔢 Total Entity Mentions", total_entity_mentions)

st.markdown("---")

# Topic Counts Bar Chart
st.markdown("## 📈 Topic Frequency Distribution")
st.markdown("This chart shows the frequency of each topic (intent) in customer queries, helping identify the most common customer support reasons.")

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
    st.markdown("This Pareto chart shows the cumulative contribution of topics to the total query volume, helping prioritize focus on the most impactful topics.")

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
    entity_df_filtered = entity_df[~entity_df["Entity Type"].isin(exclude_types)]

    # Aggregate total frequency for each entity type
    type_freq = entity_df_filtered.groupby("Entity Type")["Frequency"].sum().sort_values(ascending=False)

    st.markdown("## 🧠 Entity Type Distribution")
    st.markdown("""
    This bar chart displays the total frequency of each entity type extracted from queries.
    Note: A single query may contain multiple instances of the same entity type, 
    so the total entity mentions can exceed the total number of queries.
    For example, a query might mention multiple payment methods or dates.
    """)

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


# --- New Section: Filter queries by Entity Type and Entity Value ---
st.markdown("---")
st.markdown("## 🔍 Filter Queries by Entity Type and Entity Value")

def entity_filter_ui(entity_df, query_df):
    entity_types = sorted(entity_df['Entity Type'].unique())
    selected_entity_type = st.selectbox("Select Entity Type", entity_types)

    # Filter dataframe by selected entity type
    filtered_entities = entity_df[entity_df['Entity Type'] == selected_entity_type]

    # Map Entity Value to corresponding Query Rows (parse string to list)
    value_to_rows = {
        row['Entity Value']: ast.literal_eval(row['Query Rows']) for _, row in filtered_entities.iterrows()
    }

    selected_entity_values = st.multiselect(
        "Select Entity Value(s)",
        options=sorted(value_to_rows.keys())
    )

    if selected_entity_values:
        rows = []
        for val in selected_entity_values:
            rows.extend(value_to_rows[val])
        unique_rows = sorted(set(rows))

        st.markdown(f"**Total Queries Found:** {len(unique_rows)}")

        st.markdown("### Sample Queries containing selected entities")
        queries_to_show = []
        for r in unique_rows:
            # Assumes CSV row 2 corresponds to df index 0; adjust if needed
            idx = r - 2
            if 0 <= idx < len(query_df):
                queries_to_show.append(query_df.iloc[idx]['raw_query'])

        if queries_to_show:
            for q in queries_to_show[:30]:  # limit for UI responsiveness
                st.write(f"- {q}")
        else:
            st.info("No queries found for selected entity values.")

    else:
        st.info("Select one or more entity values to see corresponding queries.")

entity_filter_ui(entity_df, df)
