import pandas as pd
from collections import defaultdict

def analyze_bucket_keywords(df, custom_buckets):
    """
    For each keyword in the custom buckets, count number of queries it appears in,
    and map in how many unique topics the keyword appears.
    """
    keyword_topic_counts = defaultdict(lambda: defaultdict(int))
    keyword_query_counts = defaultdict(int)
    
    for _, row in df.iterrows():
        query = str(row['cleaned_query']).lower()
        topic = row['topic_label_final']
        for bucket, keywords in custom_buckets.items():
            for kw in keywords:
                # Use "in" substring checking; can be replaced by stricter matching if needed
                if kw in query:
                    keyword_topic_counts[kw][topic] += 1
                    keyword_query_counts[kw] += 1
    
    data = []
    for kw, topics in keyword_topic_counts.items():
        total_count = keyword_query_counts[kw]
        unique_topic_count = len(topics)
        data.append({
            'keyword': kw,
            'total_count': total_count,
            'unique_topic_count': unique_topic_count,
            'topics': dict(topics)
        })
    
    df_kw_summary = pd.DataFrame(data)
    
    # Sort by keywords that appear in most topics first and highest total_count descending
    df_kw_summary = df_kw_summary.sort_values(by=['unique_topic_count', 'total_count'], ascending=[False, False])
    
    return df_kw_summary

# ----- Usage Example -----
if __name__ == "__main__":
    # Load your data (adjust file path as needed)
    df = pd.read_csv("clustered_labelled_queries_final.csv")

    # Your existing custom buckets
    CUSTOM_BUCKETS = {
        "payment_method": ["credit card", "debit card", "paypal", "net banking", "upi", "wallet", "cash", "cod", "bank transfer"],
        "delivery_option": ["standard delivery", "express delivery", "same day", "next day", "pickup", "home delivery", "delivery time", "estimated delivery", "eta", "shipping", "shipping method"],
        "product": ["item", "product", "subscription", "order", "account", "profile", "invoice", "package", "plan"],
        "support_term": ["agent", "customer support", "representative", "live chat", "assistant", "chatbot", "human agent", "talk to human", "talk with someone"],
        "refund_term": ["refund", "refunded", "return", "policy", "chargeback", "cancellation", "cancellation policy", "cancel order", "cancelled", "fee", "penalty"],
        "account_action": ["reset", "register", "sign up", "login", "sign in", "verify", "delete account", "edit profile", "update info", "recover", "forgot pin", "forgot password", "switch user"],
        "tracking_info": ["track", "tracking", "order status", "shipment", "delivered", "not shipped", "delayed", "dispatched"],
        "feedback_complaint": ["feedback", "review", "complaint", "claim", "escalate", "issue", "problem", "report"],
        "newsletter": ["newsletter", "unsubscribe", "subscribe", "mailing list", "email updates"],
        "purchase_help": ["buy", "purchase", "need help purchasing", "payment failed", "cannot pay", "transaction error"]
    }

    df_kw = analyze_bucket_keywords(df, CUSTOM_BUCKETS)

    # Save output for inspection
    df_kw.to_csv("keyword_topic_analysis.csv", index=False)
    print("🔎 Keyword-topic analysis completed and saved to 'keyword_topic_analysis.csv'")
    print(df_kw.head(30))  # show top 30 by ambiguity and frequency
