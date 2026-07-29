import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import joblib

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Fake News Dashboard",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS — theme, cards, animations, transitions
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

/* ---------- Animations ---------- */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(99, 102, 241, 0.35); }
    70%  { box-shadow: 0 0 0 12px rgba(99, 102, 241, 0); }
    100% { box-shadow: 0 0 0 0 rgba(99, 102, 241, 0); }
}

/* ---------- Hero banner ---------- */
.hero {
    background: linear-gradient(120deg, #6366f1, #ec4899, #f97316);
    animation: fadeInUp 0.8s ease;
    border-radius: 20px;
    padding: 2.6rem 2rem;
    text-align: center;
    color: white;
    margin-bottom: 1.8rem;
    box-shadow: 0 10px 30px rgba(99, 102, 241, 0.25);
}
.hero h1 {
    font-family: 'Poppins', sans-serif;
    font-weight: 800;
    font-size: 2.6rem;
    margin-bottom: 0.4rem;
    text-shadow: 0 2px 10px rgba(0,0,0,0.15);
}
.hero p {
    font-size: 1.05rem;
    opacity: 0.95;
    max-width: 700px;
    margin: 0 auto;
}
.hero-icons {
    font-size: 1.8rem;
    margin-top: 0.8rem;
    letter-spacing: 0.6rem;
}

/* ---------- Metric / stat cards ---------- */
.stat-card {
    background: white;
    border-radius: 16px;
    padding: 1.3rem 1rem;
    text-align: center;
    box-shadow: 0 4px 16px rgba(0,0,0,0.06);
    border: 1px solid rgba(99,102,241,0.08);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    animation: fadeInUp 0.6s ease;
}
.stat-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 12px 28px rgba(99,102,241,0.18);
}
.stat-icon { font-size: 1.8rem; margin-bottom: 0.3rem; }
.stat-value {
    font-family: 'Poppins', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
    background: linear-gradient(90deg, #6366f1, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stat-label { color: #6b7280; font-size: 0.85rem; margin-top: 0.2rem; }

/* ---------- Section headers ---------- */
.section-title {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 1.4rem;
    margin: 1.6rem 0 0.6rem 0;
    padding-left: 0.7rem;
    border-left: 5px solid #6366f1;
    animation: fadeInUp 0.6s ease;
}

/* ---------- Prediction result cards ---------- */
.result-card {
    border-radius: 18px;
    padding: 1.6rem;
    text-align: center;
    animation: fadeInUp 0.5s ease;
    margin-top: 1rem;
}
.result-real {
    background: linear-gradient(135deg, #d1fae5, #a7f3d0);
    border: 1px solid #34d399;
}
.result-fake {
    background: linear-gradient(135deg, #fee2e2, #fecaca);
    border: 1px solid #f87171;
}
.result-card h2 { font-family:'Poppins', sans-serif; margin-bottom: 0.3rem; }

/* ---------- Buttons ---------- */
div.stButton > button {
    background: linear-gradient(90deg, #6366f1, #ec4899);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.6rem 1.4rem;
    font-weight: 600;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
div.stButton > button:hover {
    transform: scale(1.03);
    box-shadow: 0 6px 18px rgba(99,102,241,0.35);
    color: white;
}

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 10px 10px 0 0;
    padding: 8px 18px;
    font-weight: 600;
}

/* ---------- Dataframe fade-in ---------- */
[data-testid="stDataFrame"] { animation: fadeInUp 0.5s ease; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e1b4b, #312e81);
}
[data-testid="stSidebar"] * { color: #e0e7ff !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HERO BANNER
# ============================================================
st.markdown("""
<div class="hero">
    <h1>📰 Fake News Detection Dashboard</h1>
    <p>Explore a real vs. fake news dataset, visualize the patterns that separate them,
    and test your own headlines against a live machine-learning classifier.</p>
    <div class="hero-icons">🕵️ 📊 🤖 ✅ ❌</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🧭 Navigation")
    st.markdown(
        "Use the tabs at the top of the page to move between the "
        "dataset preview, visual explorations, and the live predictor."
    )
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "This dashboard trains a **TF-IDF + Logistic Regression** model "
        "to classify news articles as **Real ✅** or **Fake ❌** based on their text."
    )
    st.markdown("---")
    

# ============================================================
# LOAD DATASET (cached so it isn't reloaded on every interaction)
# ============================================================
@st.cache_data
def load_data():
    fake = pd.read_csv('Fake.csv')
    true = pd.read_csv('True.csv')

    fake['label'] = 0          # 0 = Fake
    true['label'] = 1          # 1 = Real

    df = pd.concat([fake, true], ignore_index=True)

    df['title'] = df['title'].fillna('')
    df['text'] = df['text'].fillna('')
    df['content'] = df['title'] + ' ' + df['text']

    df['text_len'] = df['text'].apply(len)
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    df['label_name'] = df['label'].map({0: 'Fake', 1: 'Real'})

    # Parse date. True.csv has trailing whitespace on some date strings which
    # breaks pandas' single-format inference, so strip first and allow mixed formats.
    df['date'] = df['date'].astype(str).str.strip()
    df['date'] = pd.to_datetime(df['date'], errors='coerce', format='mixed')
    # Drop the handful of rows (~10) where title/text/date got misaligned in the
    # source Fake.csv, so date contains a URL/junk instead of a real date.
    df = df.dropna(subset=['date']).reset_index(drop=True)

    # Shuffle so Fake/Real rows aren't in two big blocks
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df

with st.spinner("📥 Loading and preparing the dataset..."):
    df = load_data()

# ============================================================
# STAT CARDS
# ============================================================
st.markdown('<div class="section-title">📌 Dataset at a Glance</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
stat_cards = [
    (c1, "🗂️", f"{df.shape[0]:,}", "Rows"),
    (c2, "📋", f"{df.shape[1]}", "Columns"),
    (c3, "🔢", f"{df.size:,}", "Data Points"),
    (c4, "🏷️", f"{df['subject'].nunique()}", "Subjects"),
]
for col, icon, value, label in stat_cards:
    with col:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon">{icon}</div>
            <div class="stat-value">{value}</div>
            <div class="stat-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.write("")

# ============================================================
# TABS — organize the app into clear sections
# ============================================================
tab_overview, tab_explore, tab_visuals, tab_predict = st.tabs(
    ["📊 Overview", "🔍 Explore", "📈 Visualizations", "🤖 Predict"]
)

# ------------------------------------------------------------
# TAB 1 — OVERVIEW
# ------------------------------------------------------------
with tab_overview:
    st.markdown('<div class="section-title">Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(df[['title', 'text', 'subject', 'label_name']].head(), use_container_width=True)

    colA, colB, colC = st.columns(3)
    with colA:
        if st.checkbox('🧾 Column Names'):
            st.write(df.columns)
    with colB:
        if st.checkbox('📐 Statistical Summary'):
            st.write(df[['text_len', 'word_count']].describe())
    with colC:
        if st.checkbox('❓ Missing Values'):
            st.write(df.isnull().sum())

# ------------------------------------------------------------
# TAB 2 — EXPLORE
# ------------------------------------------------------------
with tab_explore:
    st.markdown('<div class="section-title">Preview by News Type</div>', unsafe_allow_html=True)
    label_choice = st.selectbox('Select News Type to Preview', df['label_name'].unique())
    st.dataframe(
        df[df['label_name'] == label_choice][['title', 'text', 'subject']].head(),
        width=True
    )

    st.markdown('<div class="section-title">Browse by Subject</div>', unsafe_allow_html=True)
    subject_choice = st.selectbox('Select Subject', sorted(df['subject'].unique()))
    st.dataframe(
        df[df['subject'] == subject_choice][['title', 'label_name']].head(),
        width=True
    )

    st.markdown('<div class="section-title">Filter by Word Count</div>', unsafe_allow_html=True)
    max_words = st.slider(
        "Select Maximum Word Count",
        int(df['word_count'].min()),
        int(df['word_count'].quantile(0.99)),   # cap at 99th percentile to avoid extreme outliers
        int(df['word_count'].quantile(0.99)),
    )
    filtered_df = df[df['word_count'] <= max_words]
    st.dataframe(filtered_df[['title', 'word_count', 'label_name']], use_container_width=True)

# ------------------------------------------------------------
# TAB 3 — VISUALIZATIONS
# ------------------------------------------------------------
with tab_visuals:
    PALETTE = ['#f87171', '#60a5fa']  # salmon-ish / sky blue, consistent across charts
    sns.set_style("whitegrid")

    st.markdown('<div class="section-title">Fake vs Real News Count</div>', unsafe_allow_html=True)
    label_counts = df['label_name'].value_counts()
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    ax.bar(label_counts.index, label_counts.values, color=PALETTE)
    ax.set_title("Count of Fake vs Real News", fontsize=9)
    ax.set_xlabel("News Type", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.tick_params(labelsize=7)
    st.pyplot(fig, use_container_width=False)

    st.markdown('<div class="section-title">News Count by Subject</div>', unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(4.5, 2.8))
    subject_counts = df.groupby(['subject', 'label_name']).size().unstack(fill_value=0)
    subject_counts.plot(kind='bar', ax=ax2, color=PALETTE)
    ax2.set_title("News Count by Subject", fontsize=9)
    ax2.set_xlabel("Subject", fontsize=8)
    ax2.set_ylabel("Count", fontsize=8)
    ax2.tick_params(labelsize=6)
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig2, use_container_width=False)

    st.markdown('<div class="section-title">Word Count Distribution</div>', unsafe_allow_html=True)
    fig3, ax3 = plt.subplots(figsize=(3.6, 2.6))
    sns.histplot(
        data=df[df['word_count'] <= df['word_count'].quantile(0.95)],
        x='word_count', hue='label_name', bins=40, ax=ax3, palette=PALETTE
    )
    ax3.set_title("Word Count Distribution (Fake vs Real)", fontsize=9)
    ax3.set_xlabel("Word Count", fontsize=8)
    ax3.tick_params(labelsize=7)
    plt.setp(ax3.get_legend().get_texts(), fontsize=7)
    st.pyplot(fig3, use_container_width=False)

    st.markdown('<div class="section-title">News Volume Over Time</div>', unsafe_allow_html=True)
    time_df = df.copy()
    time_df['month'] = time_df['date'].dt.to_period('M').dt.to_timestamp()
    monthly_counts = time_df.groupby(['month', 'label_name']).size().unstack(fill_value=0)
    fig4, ax4 = plt.subplots(figsize=(4.5, 2.8))
    monthly_counts.plot(ax=ax4, color=PALETTE)
    ax4.set_title("Monthly News Volume (Fake vs Real)", fontsize=9)
    ax4.set_xlabel("Month", fontsize=8)
    ax4.set_ylabel("Number of Articles", fontsize=8)
    ax4.tick_params(labelsize=7)
    plt.setp(ax4.get_legend().get_texts(), fontsize=7)
    st.pyplot(fig4, use_container_width=False)

# ============================================================
# TEXT CLEANING + MODEL TRAINING
# ============================================================
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))

    words = text.split()
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# Train the Model
# Problem Statement - Predict whether a news article is Fake or Real from its text
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

@st.cache_resource
def load_model():
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    accuracy = 0.9974   # Your test accuracy (99.74%)
    return model, vectorizer, accuracy

with st.spinner("🤖 Loading trained model..."):
    model, vectorizer, accuracy = load_model()

# ------------------------------------------------------------
# TAB 4 — PREDICT
# ------------------------------------------------------------
with tab_predict:

    # ============================
    # MODEL PERFORMANCE
    # ============================

    st.markdown(
        '<div class="section-title">🎯 Model Performance</div>',
        unsafe_allow_html=True
    )

    st.markdown(f"""
    <div class="stat-card" style="
        max-width:280px;
        margin-bottom:25px;">
        <div class="stat-icon">🤖</div>
        <div class="stat-value">{accuracy*100:.2f}%</div>
        <div class="stat-label">Model Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================================
# SAMPLE TEST NEWS
# ==========================================================

st.markdown(
    '<div class="section-title">📋 Sample Test News</div>',
    unsafe_allow_html=True
)

st.info("📄 Copy any sample below and paste it into the prediction box.")

if os.path.exists("Sample_Texts.txt"):

    with open("Sample_Texts.txt", "r", encoding="utf-8") as file:
        sample_text = file.read()

    st.text_area(
        label="Sample_Texts",
        value=sample_text,
        height=350,
        label_visibility="collapsed",
        key="sample_text_box"
    )

else:
    st.error("⚠ Sample_Texts.txt file not found.")

# ==========================================================
# FAKE NEWS DETECTION
# ==========================================================

st.markdown(
    '<div class="section-title">📰 Fake News Detection</div>',
    unsafe_allow_html=True
)

st.write("Paste a complete news article or headline below.")

text_input = st.text_area(
    "Paste News Article",
    height=220,
    placeholder="Paste your news article here..."
)

predict_clicked = st.button(
    "🔮 Predict News",
    use_container_width=True
)

# ==========================================================
# PREDICTION RESULT
# ==========================================================

if predict_clicked:

    if text_input.strip() == "":
        st.warning("⚠ Please enter a news article.")

    else:

        with st.spinner("🤖 Analyzing News..."):

            cleaned_text = clean_text(text_input)

            vector = vectorizer.transform([cleaned_text])

            prediction = model.predict(vector)[0]

            try:
                score = model.decision_function(vector)[0]
                confidence = (1 / (1 + np.exp(-abs(score)))) * 100
            except Exception:
                confidence = 100.0

        st.markdown(
            '<div class="section-title">📝 Prediction Result</div>',
            unsafe_allow_html=True
        )

        # Change these if your labels are reversed
        if prediction == 1:

            st.success(f"""
✅ **REAL NEWS**

**Confidence Score:** {confidence:.2f}%

The machine learning model predicts this article is **REAL / GENUINE**.
""")

        else:

            st.error(f"""
❌ **FAKE NEWS**

**Confidence Score:** {confidence:.2f}%

The machine learning model predicts this article is **FAKE / MISLEADING**.
""")

        with st.expander("🔍 View Processed Text Used by the Model"):
            st.write(cleaned_text)
