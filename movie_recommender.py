import streamlit as st
import numpy as np
import pandas as pd

# Load your data
@st.cache_data
def load_data():
    # Replace with your actual methods to load data
    df = pd.read_csv('data/movies.csv')  # Assuming a CSV file with a 'title' column
    similarity_dist = np.load('artifacts/similarity_dist.npy')  # Assuming the distance matrix is saved as a .npy file
    return df, similarity_dist

df, similarity_dist = load_data()

# Streamlit app
def recommend_movie(selected_movie_index):
    # Get similarity scores for the selected movie with all movies
    movie_distances = similarity_dist[selected_movie_index]
    
    # Set the distance to itself as infinity to ignore it
    movie_distances[selected_movie_index] = np.inf
    
    # Find the index of the movie with the smallest distance
    most_similar_movie_idx = np.argmin(movie_distances)
    
    # Return the title of the most similar movie
    return df.iloc[most_similar_movie_idx]['title']

# Title of the app and some styling
st.title('🎬 Movie Recommender System')
st.markdown("""
    <style>
    .main {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('''
    ### Find your next movie to watch!
    Select a movie you like, and we will recommend another one you might enjoy.
    ''')

sorted_movies = sorted(df['title'].unique())

# Dropdown to select a movie with some styling
selected_movie = st.selectbox('🎥 Select a movie:', sorted_movies)

# When a movie is selected
if selected_movie:
    # Find the index of the selected movie
    selected_movie_index = df.index[df['title'] == selected_movie].tolist()[0]
    
    # Get a recommendation
    recommended_movie = recommend_movie(selected_movie_index)
    
    # Display the recommended movie with enhanced text
    st.markdown(f"### 🌟 If you liked **{selected_movie}**, you might also enjoy **{recommended_movie}**!")
