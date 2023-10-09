"""
movie_recommendation.py - Um sistema de recomendação de filmes com base em títulos.
"""

import re
import numpy as np
import ipywidgets as widgets
from IPython.display import display
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Carregamento de dados
def load_data(file_path):
    """
    Carrega os dados do arquivo CSV.
    """
    return pd.read_csv(file_path)

# Limpeza de títulos
def clean_title(title):
    """
    Limpa um título removendo caracteres não alfanuméricos.
    """
    return re.sub(r"[^a-zA-Z0-9 ]", "", title)

# Pré-processamento de dados
def preprocess_data(movies):
    """
    Adiciona uma coluna 'clean_title' aos dados de filmes com títulos limpos.
    """
    movies["clean_title"] = movies["title"].apply(clean_title)
    return movies

# Cálculo do TF-IDF
def calculate_tfidf(movies):
    """
    Calcula o TF-IDF dos títulos de filmes.
    """
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf = vectorizer.fit_transform(movies["clean_title"])
    return tfidf, vectorizer

# Função para busca de filmes similares
def search(title, vectorizer, tfidf, movies):
    """
    Realiza uma pesquisa de filmes similares com base no título.
    """
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = movies.iloc[indices].iloc[::-1]
    return results

# Função para criar widget de entrada de filme
def create_movie_input_widget(default_value='Toy Story'):
    """
    Cria um widget de entrada de filme.
    """
    return widgets.Text(
        value=default_value,
        description='Movie Title:',
        disabled=False
    )


# Callback para atualização da lista de filmes
def on_movie_input_change(change):
    """
    Callback para atualizar a lista de filmes com base no título inserido.
    """
    if len(change.new) > 5:
        with movie_list:
            movie_list.clear_output()
            display(search(change.new, vectorizer, tfidf, movies))

# Função para encontrar filmes similares com base no ID do filme
def find_similar_movies(movie_id):
    """
    Encontra filmes similares com base no ID do filme.
    """
    similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)]["userId"].unique()
    similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)

    similar_user_recs = similar_user_recs[similar_user_recs > 0.10]
    all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 4)]
    all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())
    rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]

    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
    rec_percentages = rec_percentages.sort_values("score", ascending=False)
    return rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")[["score", "title", "genres"]]

# Função para criar widget de entrada de nome de filme
def create_movie_name_input_widget(default_value='Toy Story'):
    """
    Cria um widget de entrada de nome de filme.
    """
    return widgets.Text(
        value=default_value,
        description='Movie Title:',
        disabled=False
    )


# Callback para exibição de filmes recomendados
def on_movie_name_input_change(change):
    """
    Callback para exibir filmes recomendados com base no nome do filme inserido.
    """
    if len(change.new) > 5:
        results = search(change.new, vectorizer, tfidf, movies)
        if not results.empty:
            movie_id = results.iloc[0]["movieId"]
            with recommendation_list:
                recommendation_list.clear_output()

                display(find_similar_movies(movie_id))

# Carregamento de dados
movies = load_data("/workspaces/MLops2023/python_essentials_for_MLops/project1/ml-25m/movies.csv")
ratings = load_data("python_essentials_for_MLops/project1/ml-25m/ratings.csv")

# Limpeza e pré-processamento de dados
movies = preprocess_data(movies)

# Cálculo do TF-IDF
tfidf, vectorizer = calculate_tfidf(movies)

# Widgets de entrada de filme e exibição de resultados
movie_input = create_movie_input_widget()
movie_list = widgets.Output()

movie_input.observe(on_movie_input_change, names='value')

display(movie_input, movie_list)

# Widgets de entrada de nome de filme e lista de recomendações
movie_name_input = create_movie_name_input_widget()
recommendation_list = widgets.Output()

movie_name_input.observe(on_movie_name_input_change, names='value')
