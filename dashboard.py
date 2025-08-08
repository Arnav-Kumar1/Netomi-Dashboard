import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import ast
from itertools import chain
from matplotlib.ticker import FixedLocator

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
# st.sidebar.header("Filters")
# Placeholder for future filters like date ranges, topic or entity selection

# Prepare KPI metrics
valid_entity_df = entity_df[~entity_df['Entity Type'].isin(['CARDINAL', 'ORG', 'TIME'])]
all_rows = valid_entity_df['Query Rows'].dropna().apply(ast.literal_eval)
all_indices = set(chain.from_iterable(all_rows))
num_queries_with_entities = len(all_indices)
total_entity_mentions = valid_entity_df['Frequency'].sum()

# KPIs horizontally
col1, col2, col3, col4 = st.columns(4)
col1.metric("📦 Total Queries", len(df))
col2.metric("🧩 Unique Topics", df['topic_label_final'].nunique())
col3.metric("🧾 Queries With Entities", num_queries_with_entities)
col4.metric("🔢 Total Entity Mentions", total_entity_mentions)

st.markdown("---")

# Topic Frequency Bar Chart
st.markdown("## 📈 Topic Frequency Distribution")
st.markdown("This chart shows the frequency of each topic (intent) in customer queries, highlighting common customer support reasons.")

N = st.slider("Select number of top topics to display", min_value=5, max_value=len(df['topic_label_final'].unique()), value=10, key='topic_slider')

topic_counts = df['topic_label_final'].value_counts().sort_values(ascending=False)
top_N = topic_counts.head(N)

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(15, 6))  # Increased width for better spacing
ax.set_facecolor('black')
colors = plt.cm.tab20.colors
bars = ax.bar(top_N.index, top_N.values, color=colors[:len(top_N)])

# Use FixedLocator for equidistant tick positioning
ax.xaxis.set_major_locator(FixedLocator(range(len(top_N))))
ax.set_xticklabels(top_N.index, rotation=45, ha='right', fontsize=9, color='white')
ax.tick_params(axis='y', colors='white')
ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval), ha='center', va='bottom', color='white')

ax.set_title(f"Top {N} Topics by Frequency", color='white')
ax.set_ylabel("Query Count", color='white')
ax.set_xlabel("Topic", color='white')

plt.tight_layout()
st.pyplot(fig)
plt.style.use('default')

st.markdown("---")

# Pareto Chart
st.markdown("## 📉 Pareto Analysis of Topic Frequency")
st.markdown("This Pareto chart shows cumulative contribution of topics to the total query volume, helping focus on the most impactful topics.")

cumulative_percentage = topic_counts.cumsum() / topic_counts.sum() * 100

plt.style.use('dark_background')
fig2, ax1 = plt.subplots(figsize=(15, 6))  # Increased width
ax1.set_facecolor('black')
ax1.bar(topic_counts.index, topic_counts.values, color=plt.cm.Paired.colors[:len(topic_counts)])

ax1.xaxis.set_major_locator(FixedLocator(range(len(topic_counts))))
ax1.set_xticklabels(topic_counts.index, rotation=45, ha='right', fontsize=9, color='white')

ax2 = ax1.twinx()
ax2.plot(topic_counts.index, cumulative_percentage, color='cyan', marker="o")

ax2.axhline(y=80, color='gray', linestyle='--')
ax2.text(len(topic_counts) * 0.8, 82, '80% Line', color='gray')

ax1.tick_params(axis='y', colors='white')
ax2.tick_params(axis='y', colors='cyan')

for spine in ax1.spines.values():
    spine.set_color('white')
for spine in ax2.spines.values():
    spine.set_color('cyan')

ax1.set_ylabel("Query Count", color='white')
ax2.set_ylabel("Cumulative %", color='cyan')
ax1.set_title("Pareto Chart: Topic Contribution to Total Volume", color='white')

plt.tight_layout()
st.pyplot(fig2)
plt.style.use('default')

st.markdown("---")

# Entity Type Distribution Bar Chart
st.markdown("## 🧠 Entity Type Distribution")
st.markdown("""
This bar chart displays total frequency of each entity type extracted from queries.
Note: A single query may mention multiple instances of the same entity type, so total entity mentions can exceed total queries.
""")

exclude_types = {"CARDINAL", "ORG", "TIME"}
entity_df_filtered = entity_df[~entity_df["Entity Type"].isin(exclude_types)]
type_freq = entity_df_filtered.groupby("Entity Type")["Frequency"].sum().sort_values(ascending=False)

plt.style.use('dark_background')
fig3, ax = plt.subplots(figsize=(15, 6))  # Increased width for label spacing
ax.set_facecolor('black')
bars = ax.bar(type_freq.index, type_freq.values, color=plt.cm.tab20.colors[:len(type_freq)])

ax.xaxis.set_major_locator(FixedLocator(range(len(type_freq))))
ax.set_xticklabels(type_freq.index, rotation=45, ha='right', fontsize=9, color='white')
ax.tick_params(axis='y', colors='white')

for i, (etype, freq) in enumerate(type_freq.items()):
    ax.text(i, freq + 0.5, str(freq), ha='center', va='bottom', fontsize=9, color='white')

ax.set_ylabel("Total Frequency", color='white')
ax.set_xlabel("Entity Type", color='white')
ax.set_title("Entity Type Distribution (Excluding CARDINAL, ORG, TIME)", color='white')

plt.tight_layout()
st.pyplot(fig3)
plt.style.use('default')

st.markdown("---")

# Sample Queries by Topic
st.markdown("## 📝 Sample Queries by Topic")
sample_topic = st.selectbox("Select Topic for Sample Queries", sorted(df['topic_label_final'].unique()), key='sample_query_select')

group = df[df['topic_label_final'] == sample_topic]
st.markdown(f"**{sample_topic}**")

examples = group['raw_query'].dropna().sample(min(len(group), 5), random_state=42).tolist()
for q in examples:
    st.markdown(f"- {q}")

st.markdown("---")

# Filter queries by Entity Type and Entity Value
st.markdown("## 🔍 Filter Queries by Entity Type and Entity Value")

entity_types = sorted(entity_df['Entity Type'].unique())
selected_entity_type = st.selectbox("Select Entity Type", entity_types, key='entity_type_select')

filtered_entities = entity_df[entity_df['Entity Type'] == selected_entity_type]

value_to_rows = {
    row['Entity Value']: ast.literal_eval(row['Query Rows']) for _, row in filtered_entities.iterrows()
}

selected_entity_values = st.multiselect(
    "Select Entity Value(s)",
    options=sorted(value_to_rows.keys()),
    key='entity_values_select'
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
        idx = r - 2  # Adjust if Query Rows start at 2 mapping to df index 0
        if 0 <= idx < len(df):
            queries_to_show.append(df.iloc[idx]['raw_query'])

    if queries_to_show:
        for q in queries_to_show[:30]:  # limit output to 30 for UI responsiveness
            st.write(f"- {q}")
    else:
        st.info("No queries found for selected entity values.")
else:
    st.info("Select one or more entity values to see corresponding queries.")
