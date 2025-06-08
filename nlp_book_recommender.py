import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
@st.cache_data

def load_data():
    df = pd.read_csv("data/books.csv")
    df['description'] = df['description'].fillna("")
    return df

def build_recommender(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['description'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    title_to_index = pd.Series(df.index, index=df['title']).drop_duplicates()
    return cosine_sim, title_to_index

def recommend(title, df, cosine_sim, title_to_index, num_recommendations=5):
    idx = title_to_index.get(title)
    if idx is None:
        return ["Book not found."]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    book_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[book_indices].tolist()

# Streamlit UI
st.set_page_config(page_title="📚 Book Recommender")
st.title("📚 NLP Book Recommender")
st.markdown("Enter a book title to get similar recommendations.")

# Load and process data
df = load_data()
cosine_sim, title_to_index = build_recommender(df)

# Input box
book_title = st.text_input("Enter a book title", "The Great Gatsby")

# Recommend button
if st.button("Recommend"):
    recommendations = recommend(book_title, df, cosine_sim, title_to_index)
    if recommendations[0] == "Book not found.":
        st.error("Book not found. Try another title.")
    else:
        st.success(f"Because you liked '{book_title}':")
        for r in recommendations:
            st.write(f"- {r}")
