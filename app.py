
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
DATA = BASE / "data"

st.set_page_config(page_title="Doomsday Trailer Opinion Mining", layout="wide")
st.title("Avengers: Doomsday Trailer — Opinion Mining Dashboard")
st.caption("Sentiment classification, TF-IDF, dimensionality reduction, and tag generation")

def load_csv(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

comments = load_csv(OUT / "analyzed_comments.csv")
bow = load_csv(OUT / "bow_word_frequencies.csv")
tfidf = load_csv(OUT / "tfidf_scores.csv")
ranked = load_csv(OUT / "ranked_tags.csv")
refined = load_csv(OUT / "refined_tags.csv")
enriched = load_csv(OUT / "enriched_tags.csv")

if comments.empty:
    st.warning("No analysis outputs found. Run the notebook analysis cells first.")
    st.stop()

total = len(comments)
positive = int((comments["sentiment"] == "Positive").sum())
neutral = int((comments["sentiment"] == "Neutral").sum())
negative = int((comments["sentiment"] == "Negative").sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Comments analyzed", f"{total:,}")
c2.metric("Positive", f"{positive:,}", f"{positive / total:.1%}")
c3.metric("Neutral", f"{neutral:,}", f"{neutral / total:.1%}")
c4.metric("Negative", f"{negative:,}", f"{negative / total:.1%}")

left, right = st.columns(2)
with left:
    sentiment = comments["sentiment"].value_counts().rename_axis("sentiment").reset_index(name="count")
    st.plotly_chart(px.pie(sentiment, names="sentiment", values="count", hole=0.4,
                           title="Sentiment Distribution"), use_container_width=True)
with right:
    st.plotly_chart(px.bar(sentiment, x="sentiment", y="count", color="sentiment",
                           title="Sentiment Counts", text="count"), use_container_width=True)

st.subheader("Bag of Words — Most Frequent Words")
if not bow.empty:
    plot_data = bow.head(20).sort_values("frequency")
    st.plotly_chart(px.bar(plot_data, x="frequency", y="word", orientation="h",
                           title="Top 20 Words"), use_container_width=True)
    st.dataframe(bow.head(30), use_container_width=True)

st.subheader("TF-IDF — Most Important Terms")
if not tfidf.empty:
    plot_data = tfidf.head(20).sort_values("mean_tfidf")
    st.plotly_chart(px.bar(plot_data, x="mean_tfidf", y="term", orientation="h",
                           title="Top 20 Terms by Mean TF-IDF"), use_container_width=True)
    st.dataframe(tfidf.head(30), use_container_width=True)

st.subheader("Dimensionality Reduction — SVD")
if {"svd_1", "svd_2"}.issubset(comments.columns):
    st.plotly_chart(px.scatter(comments, x="svd_1", y="svd_2", color="sentiment",
                               hover_data=["comment"], title="2D SVD Projection"),
                    use_container_width=True)

st.subheader("Tag Generation")
tab1, tab2, tab3 = st.tabs(["Ranked tags", "Refined tags", "Enriched tags"])
with tab1:
    if not ranked.empty:
        st.dataframe(ranked, use_container_width=True)
        weights = dict(zip(ranked["term"].astype(str), ranked["mean_tfidf"].astype(float)))
        if weights:
            cloud = WordCloud(width=1100, height=450, background_color="white",
                              collocations=False).generate_from_frequencies(weights)
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.imshow(cloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
with tab2:
    st.dataframe(refined, use_container_width=True)
with tab3:
    st.dataframe(enriched, use_container_width=True)

st.subheader("Comment-Level Results")
sentiments = ["All"] + sorted(comments["sentiment"].dropna().unique().tolist())
selected = st.selectbox("Filter by sentiment", sentiments)
filtered = comments if selected == "All" else comments[comments["sentiment"] == selected]
search = st.text_input("Search comments")
if search.strip():
    filtered = filtered[filtered["comment"].str.contains(search, case=False, na=False)]
cols = [c for c in ["comment", "clean_text", "sentiment", "sentiment_score", "like_count", "published_at"] if c in filtered.columns]
st.dataframe(filtered[cols], use_container_width=True, height=420)

st.download_button("Download analyzed comments CSV",
                   data=comments.to_csv(index=False).encode("utf-8"),
                   file_name="analyzed_comments.csv", mime="text/csv")
