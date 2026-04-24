import streamlit as st
import sqlite3
import pandas as pd
import re
from datetime import datetime
from arabic_nlp_pipeline import ArabicSentimentAnalyzer
from transformers import pipeline

# --- Page Configuration ---
st.set_page_config(page_title="Arabic & English NLP Dashboard", page_icon="🌍", layout="wide")

# --- Database Initialization & Migration ---
def init_db():
    conn = sqlite3.connect('sentiment_tracking.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            original_text TEXT,
            sentiment TEXT,
            confidence TEXT,
            top_emotion TEXT
        )
    ''')
    
    # Graceful Migration: Add 'language' column if it doesn't exist yet
    c.execute("PRAGMA table_info(predictions)")
    columns = [col[1] for col in c.fetchall()]
    if 'language' not in columns:
        c.execute("ALTER TABLE predictions ADD COLUMN language TEXT DEFAULT 'Unknown'")
        
    conn.commit()
    return conn

def save_prediction(conn, text, sentiment, conf, emotion, language):
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''
        INSERT INTO predictions (timestamp, original_text, sentiment, confidence, top_emotion, language)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (timestamp, text, sentiment, conf, emotion, language))
    conn.commit()

def load_history(conn):
    return pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC LIMIT 50", conn)

# --- Language Detection Helper ---
def detect_language(text):
    """Simple detector based on character sets."""
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', str(text)))
    english_chars = len(re.findall(r'[a-zA-Z]', str(text)))
    return "en" if english_chars > arabic_chars else "ar"

# --- Unified Predictor ---
def analyze_mixed_language(text, arabic_analyzer, english_sentiment_model):
    lang = detect_language(text)
    
    if lang == "ar":
        res = arabic_analyzer.predict(text)
        res['language'] = "Arabic 🇪🇬"
        return res
    else:
        # Route to English Sentiment Pipeline
        sent_result = english_sentiment_model(text)[0]
        label_map = {"POSITIVE": "Positive 😊", "NEGATIVE": "Negative 😡"}
        sentiment = label_map.get(sent_result['label'], "Neutral 😐")
        
        # Reuse XLM-RoBERTa (which is multilingual) for Emotion intent
        en_labels = ["happiness", "anger", "sadness", "sarcasm", "complaint"]
        emo_result = arabic_analyzer.emotion_classifier(text, en_labels)
        top_emotion = emo_result['labels'][0].capitalize()
        
        return {
            "original": text,
            "cleaned": text.strip(),
            "sentiment": sentiment,
            "confidence": f"{sent_result['score']:.2%}",
            "top_emotion": top_emotion,
            "emotion_confidence": f"{emo_result['scores'][0]:.2%}",
            "language": "English 🇬🇧"
        }

# --- Batch Analysis Helpers ---
def get_top_reasons(texts, stop_words, top_n=5):
    from sklearn.feature_extraction.text import CountVectorizer
    if not texts:
        return []
    vectorizer = CountVectorizer(ngram_range=(2, 3), stop_words=stop_words, min_df=2)
    try:
        X = vectorizer.fit_transform(texts)
        phrase_frequencies = X.sum(axis=0).A1
        phrases = vectorizer.get_feature_names_out()
        phrase_counts = list(zip(phrases, phrase_frequencies))
        phrase_counts.sort(key=lambda x: x[1], reverse=True)
        return phrase_counts[:top_n]
    except ValueError:
        return [("Not enough data to find trends", 0)]

# --- Load NLP Models ---
@st.cache_resource
def load_models():
    # Load custom Arabic pipeline
    arabic_analyzer = ArabicSentimentAnalyzer()
    # Load lightweight default English pipeline
    english_sentiment = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
    return arabic_analyzer, english_sentiment

# --- Main App ---
def main():
    st.title("🌍 Multilingual Sentiment & Emotion Analyzer")
    st.markdown("Automatically detects and analyzes English and Egyptian Arabic customer feedback.")

    conn = init_db()
    
    with st.spinner('Loading Language Models... (This takes a moment on first run)'):
        arabic_analyzer, english_sentiment = load_models()

    tab1, tab2 = st.tabs(["💬 Single Review Analysis", "📊 Batch CSV Analysis"])

    with tab1:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Analyze Text")
            user_input = st.text_area("Enter a tweet or review (English, MSA, or Egyptian Arabic):", height=150, 
                                      placeholder="مثال: أسوأ خدمة عملاء شفتها... OR Example: Terrible customer service...")
            
            if st.button("Analyze Sentiment", type="primary"):
                if user_input.strip():
                    with st.spinner("Detecting language and analyzing text..."):
                        res = analyze_mixed_language(user_input, arabic_analyzer, english_sentiment)
                        
                        # Display metrics
                        st.success("Analysis Complete!")
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Language", res['language'])
                        m2.metric("Polarity", res['sentiment'])
                        m3.metric("Emotion / Intent", res['top_emotion'])
                        m4.metric("Confidence", res['confidence'])
                        
                        st.info(f"**Cleaned Text:** {res['cleaned']}")
                        
                        # Save to DB
                        save_prediction(conn, res['original'], res['sentiment'], res['confidence'], res['top_emotion'], res['language'])
                else:
                    st.warning("Please enter some text to analyze.")

        with col2:
            st.subheader("Recent History")
            st.markdown("Saved automatically to local SQLite database.")
            df = load_history(conn)
            
            if not df.empty:
                st.dataframe(df[['original_text', 'language', 'sentiment', 'top_emotion']], use_container_width=True, hide_index=True)
            else:
                st.write("No history available yet.")

    with tab2:
        st.subheader("Batch Dataset Analyzer")
        st.markdown("Upload a CSV file containing customer reviews. The AI will handle mixed English/Arabic datasets automatically.")
        
        uploaded_file = st.file_uploader("Upload CSV Dataset", type=['csv'])
        if uploaded_file is not None:
            df_batch = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df_batch.head(3))
            
            text_column = st.selectbox("Select the column containing the review text:", df_batch.columns)
            
            if st.button("Start Batch Analysis", type="primary", key="batch_btn"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = []
                texts_to_analyze = df_batch[text_column].dropna().astype(str).tolist()
                total_rows = len(texts_to_analyze)
                
                for i, text in enumerate(texts_to_analyze):
                    res = analyze_mixed_language(text, arabic_analyzer, english_sentiment)
                    results.append({
                        "original_text": text,
                        "cleaned_text": res['cleaned'],
                        "language": res['language'],
                        "sentiment": res['sentiment'],
                        "top_emotion": res['top_emotion']
                    })
                    
                    if i % 5 == 0 or i == total_rows - 1:
                        progress_bar.progress((i + 1) / total_rows)
                        status_text.text(f"Analyzing review {i + 1} of {total_rows}...")
                        
                status_text.success("Batch Analysis Complete!")
                results_df = pd.DataFrame(results)
                
                # Analytics
                st.markdown("### 📊 Overall Sentiment Distribution")
                sentiment_counts = results_df['sentiment'].value_counts(normalize=True) * 100
                st.bar_chart(sentiment_counts)
                
                # Combined Stop Words for English and Arabic extraction
                combined_stop_words = [
                    "في", "من", "على", "عن", "مع", "اللي", "ده", "دي", "انا", "هو", "هي", "احنا", "كان", "كانت", "ان", "لو", "يا", "بس", "عشان", "علشان", "لما", "ولا", "او", "اي", "ايه", "كده", "جدا", "والله", "طب", "مش", "ما",
                    "the", "is", "in", "and", "to", "it", "that", "of", "for", "on", "this", "with", "was", "as", "are", "be", "have", "you", "they", "we", "not", "but"
                ]
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("#### 🚨 Top Reasons for Negative Reviews")
                    neg_texts = results_df[results_df['sentiment'].str.contains('Negative', na=False)]['cleaned_text'].tolist()
                    neg_reasons = get_top_reasons(neg_texts, combined_stop_words)
                    for phrase, count in neg_reasons:
                        st.error(f"**{phrase}** ({count} mentions)")
                        
                with c2:
                    st.markdown("#### ⭐ Top Reasons for Positive Reviews")
                    pos_texts = results_df[results_df['sentiment'].str.contains('Positive', na=False)]['cleaned_text'].tolist()
                    pos_reasons = get_top_reasons(pos_texts, combined_stop_words)
                    for phrase, count in pos_reasons:
                        st.success(f"**{phrase}** ({count} mentions)")
                        
                st.markdown("### 📥 Download Results")
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Full Analyzed Results (CSV)",
                    data=csv,
                    file_name='multilingual_batch_results.csv',
                    mime='text/csv',
                )

if __name__ == "__main__":
    main()