import pandas as pd
import numpy as np
import pytest
from gensim.models import Word2Vec
import pickle
import os

# Import the function
from projetofinal.preprocessing import (
    map_dict,
    encode_target,
    compute_avg_embedding,
    embbeded_model,
    embedded_data,
    save_as_pickle,
)


# Define a fixture to generate sample data
@pytest.fixture
def sample_data1():
    data = {
        "TerritorioCon": ["A", "B", "C", "A", "B"],
        "SetorInstitucionalCon": ["X", "Y", "Z", "X", "Y"],
    }
    return pd.DataFrame(data)


# Define the test function
def test_map_dict(sample_data1):
    # Call the function with sample data
    df_clean, inverted_mapping_ter, inverted_mapping_sec = map_dict(sample_data1)

    # Assert that the returned values are as expected
    assert "encoded_label_territorio" in df_clean.columns
    assert "encoded_label_setor" in df_clean.columns
    assert isinstance(inverted_mapping_ter, dict)
    assert isinstance(inverted_mapping_sec, dict)


def test_encode_target():
    # Define a sample category mapping
    category_mapping = {"A": 0, "B": 1}

    assert encode_target("A", category_mapping) == 0
    assert encode_target("C", category_mapping) == 2
    assert category_mapping["C"] == 2


class MockWord2VecModel:
    def __init__(self):
        self.vector_size = 100
        self.wv = {"apple": np.ones(100), "banana": np.ones(100) * 2}


# Define a test function
def test_compute_avg_embedding():
    # Create a sample Word2Vec model
    word2vec_model = MockWord2VecModel()

    # Test when tokens contain embeddings
    tokens_with_embeddings = ["apple", "banana"]
    avg_embedding = compute_avg_embedding(tokens_with_embeddings, word2vec_model)
    assert np.array_equal(avg_embedding, np.ones(100) * 1.5)

    # Test when tokens do not contain embeddings
    tokens_without_embeddings = ["orange", "grape"]
    avg_embedding = compute_avg_embedding(tokens_without_embeddings, word2vec_model)
    assert np.array_equal(avg_embedding, np.zeros(100))

    # Test with an empty list of tokens
    avg_embedding = compute_avg_embedding([], word2vec_model)
    assert np.array_equal(avg_embedding, np.zeros(100))


@pytest.fixture
def sample_data2():
    data = {
        "DescricaoInstrumento": [
            "This is a test description.",
            "Another description here.",
        ]
    }
    return pd.DataFrame(data)


# Define a test function
def test_embbeded_model(sample_data2):
    # Call the function with sample data
    df_clean, word2vec_model = embbeded_model(sample_data2)

    # Check if the additional column 'tokenized_Descricao_text' is added
    assert "tokenized_Descricao_text" in df_clean.columns

    # Check if the Word2Vec model is trained
    assert isinstance(word2vec_model, Word2Vec)

    # Check if the Word2Vec model contains the vocabulary
    assert "test" in word2vec_model.wv
    assert "description" in word2vec_model.wv
    assert "another" in word2vec_model.wv


@pytest.fixture
def sample_data3():
    data = {
        "tokenized_Descricao_text": [
            ["this", "is", "a", "test"],
            ["another", "description"],
        ],
        "encoded_label_territorio": [0, 1],
        "encoded_label_setor": [2, 3],
    }
    return pd.DataFrame(data)


# Define a test function
def test_embedded_data(sample_data3):
    # Create a sample Word2Vec model
    word2vec_model = MockWord2VecModel()

    # Call the function with sample data and Word2Vec model
    df_clean, X, y1, y2 = embedded_data(sample_data3, word2vec_model)

    # Check if the additional column 'avg_embedding' is added
    assert "avg_embedding" in df_clean.columns

    # Check the dimensions of X
    assert X.shape == (2, 100)

    # Check the values of y1 and y2
    assert (y1 == [0, 1]).all()
    assert (y2 == [2, 3]).all()
