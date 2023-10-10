# Importe as bibliotecas necessárias
import pytest
import pandas as pd
from movie_recommendation import load_data, clean_title, preprocess_data, calculate_tfidf, search, find_similar_movies

def test_load_data():
    data = load_data("test_data.csv")
    assert data is not None

def test_clean_title():
    title = "Avatar (2009)"
    cleaned_title = clean_title(title)
    assert cleaned_title == "Avatar 2009"

def test_preprocess_data():
    data = pd.DataFrame({"title": ["Avatar (2009)"]})
    processed_data = preprocess_data(data)
    assert "clean_title" in processed_data.columns

def test_calculate_tfidf():
    data = pd.DataFrame({"title": ["Avatar (2009)"]})
    tfidf, vectorizer = calculate_tfidf(data)
    assert tfidf is not None
    assert vectorizer is not None

def test_search():
    data = pd.DataFrame({"title": ["Avatar (2009)"]})
    tfidf, vectorizer = calculate_tfidf(data)
    movies = preprocess_data(data)
    results = search("Avatar (2009)", vectorizer, tfidf, movies)
    assert not results.empty

def test_find_similar_movies():
    movie_id = 1
    similar_movies = find_similar_movies(movie_id)
    assert not similar_movies.empty
