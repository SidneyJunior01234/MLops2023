import pandas as pd
import numpy as np
from unittest.mock import patch
import os
import pytest
from Credit_Card_Customer_Segmentation import (
    load_customer_data,
    explore_categorical_data,
    preprocess_data,
    scale_data,
    find_optimal_clusters,
    cluster_data,
)

TEST_DATA_DIR = "test_data"

TEST_DATA_FILE = os.path.join(TEST_DATA_DIR, "test_customer_segmentation.csv")

TEST_DATA = pd.DataFrame(
    {
        "customer_id": [1, 2, 3],
        "gender": ["M", "F", "M"],
        "education_level": ["Graduate", "High School", "College"],
        "marital_status": ["Married", "Single", "Single"],
    }
)

@pytest.fixture
def create_test_data_csv():
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    TEST_DATA.to_csv(TEST_DATA_FILE, index=False)

@pytest.fixture
def clean_test_data_csv():
    os.remove(TEST_DATA_FILE)

def test_load_customer_data(create_test_data_csv, clean_test_data_csv):

    df = load_customer_data(TEST_DATA_FILE)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3

    with pytest.raises(Exception):
        df = load_customer_data("invalid_file.csv")

def test_explore_categorical_data(create_test_data_csv, clean_test_data_csv, capsys):
    df = load_customer_data(TEST_DATA_FILE)
    categorical_columns = ["gender", "education_level", "marital_status"]

    explore_categorical_data(df, categorical_columns)
    captured = capsys.readouterr()
    assert "gender" in captured.out
    assert "education_level" in captured.out
    assert "marital_status" in captured.out

    with patch("sys.stdout", side_effect=IOError()):
        explore_categorical_data(df, categorical_columns)

def test_preprocess_data(create_test_data_csv, clean_test_data_csv):
    df = load_customer_data(TEST_DATA_FILE)

    preprocessed_df = preprocess_data(df)
    assert "gender" in preprocessed_df.columns
    assert "education_level" not in preprocessed_df.columns
    assert "marital_status" not in preprocessed_df.columns

    with pytest.raises(Exception):
        preprocessed_df = preprocess_data(None)

def test_scale_data(create_test_data_csv, clean_test_data_csv):
    df = load_customer_data(TEST_DATA_FILE)

    scaled_df = scale_data(df.drop("customer_id", axis=1))
    assert isinstance(scaled_df, pd.DataFrame)
    assert len(scaled_df) == 3

    with pytest.raises(Exception):
        scaled_df = scale_data(None)

def test_find_optimal_clusters(create_test_data_csv, clean_test_data_csv):
    df = load_customer_data(TEST_DATA_FILE)

    max_clusters = 5
    inertias = find_optimal_clusters(df.drop("customer_id", axis=1), max_clusters)
    assert isinstance(inertias, list)
    assert len(inertias) == max_clusters

    with pytest.raises(Exception):
        inertias = find_optimal_clusters(None)

def test_cluster_data(create_test_data_csv, clean_test_data_csv):
    df = load_customer_data(TEST_DATA_FILE)
    scaled_df = scale_data(df.drop("customer_id", axis=1))

    num_clusters = 3
    cluster_labels = cluster_data(scaled_df, num_clusters)
    assert isinstance(cluster_labels, np.ndarray)
    assert len(cluster_labels) == len(df)

    with pytest.raises(Exception):
        cluster_labels = cluster_data(None, num_clusters)
