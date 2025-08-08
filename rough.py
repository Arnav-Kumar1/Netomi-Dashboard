import pandas as pd
from collections import defaultdict

def analyze_candidate_phrases_for_buckets(df, existing_buckets):
    """
    Find frequent words and two-word phrases in queries not in existing buckets,
    to suggest candidates for new entity buckets.
    """
    existing_keywords = set()
    for keywords in existing_buckets.values():
        for kw in keywords:
            existing_keywords.add(kw.lower())

    word_counts = defaultdict(int)
    phrase_counts = defaultdict(int)

    for q in df['cleaned_query'].dropna():
        query = q.lower()
        words = query.split()

        for word in words:
            if word not in existing_keywords:
                word_counts[word] += 1

        bigrams = [" ".join(words[i:i+2]) for i in range(len(words)-1)]
        for bg in bigrams:
            if bg not in existing_keywords:
                phrase_counts[bg] += 1

    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:30]
    sorted_phrases = sorted(phrase_counts.items(), key=lambda x: x[1], reverse=True)[:30]

    return sorted_words, sorted_phrases

# Load your dataframe and set your current custom buckets dictionary
df = pd.read_csv("clustered_labelled_queries_final.csv")

REFINED_CUSTOM_BUCKETS = {
    "payment_method": ["credit card", "debit card", "paypal", "net banking", "upi", "wallet", "cash", "cod", "bank transfer"],
    "delivery_option": ["standard delivery", "express delivery", "pickup", "home delivery", "delivery time", "estimated delivery", "eta", "shipping method"],
    "delivery_address": ["shipping address", "delivery address", "new address", "change address", "set address"],
    "order_reference": ["cancel order", "order status", "track order", "modify order", "order number"],
    "invoice_reference": ["invoice", "invoice number", "last invoice", "check invoice"],
    "account_action": ["account deletion", "account recovery", "account creation", "edit profile", "reset password", "register", "sign up"],
    "refund_request": ["refund request", "refund policy", "cancellation policy", "cancelled", "cancellation fee"],
    "support_request": ["help me", "need help", "assistance", "can you help", "help to", "you help"],
    "status_check": ["check", "to check", "need to check", "check status", "check refund", "check order"],
    "recipient_person": ["my mom", "my wife", "my dad", "my daughter"],
    "support_channel": ["agent", "customer support", "live chat", "human agent", "talk to human"],
    "newsletter_action": ["newsletter", "subscribe", "unsubscribe", "mailing list"],
    "purchase_help": ["buy", "purchase", "payment failed", "transaction error", "cannot pay"],
    "instruction_request": ["how to", "know how", "can i", "could you", "know what"],
    "general_complaint": ["problem", "issue", "report", "complaint", "feedback", "claim", "escalate"],
}

words_not_covered, phrases_not_covered = analyze_candidate_phrases_for_buckets(df, CURRENT_CUSTOM_BUCKETS)

print("\nTop 30 frequent words outside current buckets:")
for word, count in words_not_covered:
    print(f"{word}: {count}")

print("\nTop 30 frequent two-word phrases outside current buckets:")
for phrase, count in phrases_not_covered:
    print(f"{phrase}: {count}")
