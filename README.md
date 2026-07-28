# 📰 Fake News Detection Dashboard

An interactive Streamlit dashboard that explores a real-vs-fake news dataset and
lets you test your own headlines/articles against a live machine-learning
classifier (TF-IDF + Logistic Regression).

## ✨ Features

- **Dataset overview** — row/column counts, column names, statistical summary,
  missing values.
- **Explore tab** — filter articles by news type, subject, and word count.
- **Visualizations tab** — Fake vs Real counts, news volume by subject, word
  count distribution, and monthly volume over time.
- **Predict tab** — paste in any headline or article text and get a
  Real ✅ / Fake ❌ prediction with a confidence score.
- Custom styled UI with gradient hero banner, animated stat cards, and a
  dark sidebar.
- Trained model is cached to disk (`model_cache.joblib`) so it only needs to
  train once — every app restart after that loads instantly.

## 🗂️ Project Structure

```
## 🗂️ Project Structure

```text
CAPSTONE - PROJECT/
│
├── app.py                        # Main Streamlit application
├── stapp.py                      # Alternative Streamlit app (if used)
│
├── Fake.csv                      # Fake news dataset
├── True.csv                      # Real news dataset
├── cleaned_fake_news.csv         # Cleaned dataset
├── processed_fake_news.csv       # Processed dataset
├── final_processed_news.csv      # Final processed dataset
│
├── fake_news_model.pkl           # Trained Machine Learning model
├── tfidf_vectorizer.pkl          # Saved TF-IDF vectorizer
├── model_cache.joblib            # Cached model for faster loading
│
├── FAKENEWS.ipynb                # Model training notebook
├── Business Queries.ipynb        # Business queries & EDA notebook
│
├── Sample_Texts.txt              # Sample news articles for testing
│
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
```

## 📦 Requirements

- Python 3.9+
- streamlit
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib

Install everything with:

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn joblib
```

## 📁 Dataset

This app expects two CSV files in the **same folder as `app.py`**:

- `Fake.csv`
- `True.csv`

Both need at minimum these columns: `title`, `text`, `subject`, `date`.
(This matches the popular "Fake and Real News Dataset" commonly found on
Kaggle.)

## ▶️ How to Run

1. Place `Fake.csv` and `True.csv` in the project folder.
2. Install the requirements above.
3. From the project folder, run:

   ```bash
   streamlit run stapp.py
   ```

4. Your browser should open automatically at `http://localhost:8501`.
   If not, open that URL manually.

**First run note:** the very first launch will take a bit longer while the
model trains on ~45,000 articles. After that, it loads from
`model_cache.joblib` and starts instantly — delete that file if you ever
want to force a retrain (e.g. after changing the dataset).

## 🔮 Using the Predict Tab

Go to the **🤖 Predict** tab, paste in a news headline or article body, and
click **Predict**. You'll get:

- A **Real ✅** or **Fake ❌** label
- A confidence percentage

Check `sample_texts.txt` for a few ready-made examples you can copy in to
try it out quickly.

## ⚠️ Notes & Limitations

- The model is trained only on the dataset provided — predictions on topics,
  writing styles, or time periods very different from that dataset may be
  less reliable.
- This is a demo/educational tool, not a fact-checking service. A "Real"
  prediction does not verify factual accuracy, and a "Fake" prediction is
  not a legal or journalistic determination.

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) — UI framework
- [scikit-learn](https://scikit-learn.org/) — TF-IDF vectorizer + Logistic
  Regression classifier
- [pandas](https://pandas.pydata.org/) / [numpy](https://numpy.org/) — data
  handling
- [matplotlib](https://matplotlib.org/) / [seaborn](https://seaborn.pydata.org/)
  — charts
