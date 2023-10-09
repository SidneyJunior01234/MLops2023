"""
movie_recommendation.py - Um sistema de recomendação de filmes com base em títulos.
"""

import re
import logging
import numpy as np
import ipywidgets as widgets
from IPython.display import display
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(filename='movie_recommendation.log', level=logging.ERROR)

def load_data(file_path):
    """
    Carrega os dados do arquivo CSV.
    """
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        logging.error("Erro no carregamento de dados: %s", str(e))
        return None

def clean_title(title):
    """
    Limpa um título removendo caracteres não alfanuméricos.
    """
    return re.sub(r"[^a-zA-Z0-9 ]", "", title)

def preprocess_data(movies_data):
    """
    Adiciona uma coluna 'clean_title' aos dados de filmes com títulos limpos.
    """
    try:
        movies_data["clean_title"] = movies_data["title"].apply(clean_title)
        return movies_data
    except Exception as e:
        logging.error("Erro no pré-processamento de dados: %s", str(e))
        return None


def calculate_tfidf(movies_data):
    """
    Calcula o TF-IDF dos títulos de filmes.
    """
    try:
        local_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        tfidf_data = local_vectorizer .fit_transform(
            movies_data["clean_title"])
        return tfidf_data, local_vectorizer
    except Exception as e:
        logging.error("Erro no cálculo do TF-IDF: %s", str(e))
        return None, None


def search(title, vectorizer, tfidf, movies):
    """
    Realiza uma pesquisa de filmes similares com base no título.
    """
    try:
        title = clean_title(title)
        query_vec = vectorizer.transform([title])
        similarity = cosine_similarity(query_vec, tfidf).flatten()
        indices = np.argpartition(similarity, -5)[-5:]
        results = movies.iloc[indices].iloc[::-1]
        return results
    except Exception as e:
        logging.error("Erro na busca de filmes similares: %s", str(e))
        return None


def create_movie_input_widget(default_value='Toy Story'):
    """
    Cria um widget de entrada de filme.
    """
    try:
        return widgets.Text(
            value=default_value,
            description='Movie Title:',
            disabled=False
        )
    except Exception as e:
        logging.error("Erro ao criar widget de entrada de filme: %s", str(e))
        return None

def on_movie_input_change(change):
    """
    Callback para atualizar a lista de filmes com base no título inserido.
    """
    try:
        if len(change.new) > 5:
            with movie_list:
                movie_list.clear_output()
                display(search(change.new, vectorizer, tfidf, movies))
    except ValueError as ve:
        logging.error("Erro na atualização da lista de filmes: %s", str(ve))
    except Exception as e:
        logging.error(
            "Erro desconhecido na atualização da lista de filmes: %s", str(e))

def find_similar_movies(movie_id):
    """
    Encontra filmes similares com base no ID do filme.
    """
    try:
        similar_users = ratings[(ratings["movieId"] == movie_id)
                                & (ratings["rating"] > 4)]["userId"].unique()
        similar_user_recs = ratings[(ratings["userId"].isin(similar_users))
                                    & (ratings["rating"] > 4)]["movieId"]
        similar_user_recs = similar_user_recs.value_counts() / len(similar_users)
        similar_user_recs = similar_user_recs[similar_user_recs > 0.10]
        all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index))
                            & (ratings["rating"] > 4)]
        all_user_recs = all_users["movieId"].value_counts(
        ) / len(all_users["userId"].unique())
        rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
        rec_percentages.columns = ["similar", "all"]
        rec_percentages["score"] = rec_percentages["similar"] / \
            rec_percentages["all"]
        rec_percentages = rec_percentages.sort_values("score", ascending=False)
        return rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")[["score", "title", "genres"]]
    except KeyError as ke:
        logging.error(
            "Erro ao encontrar filmes similares - KeyError: %s", str(ke))
    except Exception as e:
        logging.error(
            "Erro desconhecido ao encontrar filmes similares: %s", str(e))

def create_movie_name_input_widget(default_value='Toy Story'):
    """
    Cria um widget de entrada de nome de filme.
    """
    try:
        return widgets.Text(
            value=default_value,
            description='Movie Title:',
            disabled=False
        )
    except Exception as e:
        logging.error(
            "Erro ao criar widget de entrada de nome de filme: %s", str(e))
        return None

def on_movie_name_input_change(change):
    """
    Callback para exibir filmes recomendados com base no nome do filme inserido.
    """
    try:
        if len(change.new) > 5:
            results = search(change.new, vectorizer, tfidf, movies)
            if not results.empty:
                movie_id = results.iloc[0]["movieId"]
                with recommendation_list:
                    recommendation_list.clear_output()
                    display(find_similar_movies(movie_id))
    except ValueError as ve:
        logging.error("Erro na exibição de filmes recomendados: %s", str(ve))
    except Exception as e:
        logging.error("Erro desconhecido na exibição de filmes recomendados: %s", str(e))

load_movies_data = load_data(
    "/workspaces/MLops2023/python_essentials_for_MLops/project1/ml-25m/movies.csv")
ratings = load_data("python_essentials_for_MLops/project1/ml-25m/ratings.csv")

if load_movies_data is not None and ratings is not None:

    movies = preprocess_data(load_movies_data)

    tfidf, vectorizer = calculate_tfidf(movies)
    if tfidf is not None and vectorizer is not None:

        movie_input = create_movie_input_widget()
        movie_list = widgets.Output()
        movie_input.observe(on_movie_input_change, names='value')
        display(movie_input, movie_list)

        movie_name_input = create_movie_name_input_widget()
        recommendation_list = widgets.Output()
        movie_name_input.observe(on_movie_name_input_change, names='value')
else:
    print("Erro no carregamento de dados. Verifique o arquivo de log para mais detalhes.")
