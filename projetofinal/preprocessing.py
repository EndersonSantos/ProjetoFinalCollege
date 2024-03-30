from gensim.utils import simple_preprocess
from gensim.models import Word2Vec
import numpy as np
import pandas  as pd
import pickle


def map_dict(df_clean):
    """
    Maps unique categories in the 'TerritorioCon' and 'SetorInstitucionalCon' columns
    of a DataFrame to integer labels and returns the encoded DataFrame along with
    inverted mappings for both columns.

    Parameters:
        df_clean (DataFrame): DataFrame containing the columns 'TerritorioCon' and 'SetorInstitucionalCon'.

    Returns:
        DataFrame: DataFrame with additional columns 'encoded_label_territorio' and 'encoded_label_setor'.
        dict: Inverted mapping for 'TerritorioCon' column.
        dict: Inverted mapping for 'SetorInstitucionalCon' column.
    """
    unique_categories_ter = df_clean['TerritorioCon'].unique()
    category_mapping_ter = dict(zip(unique_categories_ter, range(len(unique_categories_ter))))
    inverted_mapping_ter = {value: key for key, value in category_mapping_ter.items()}

    unique_categories_sec = df_clean["SetorInstitucionalCon"].unique()
    category_mapping_sec = dict(zip(unique_categories_sec, range(len(unique_categories_sec))))
    inverted_mapping_sec = {value: key for key, value in category_mapping_sec.items()}

    df_clean["encoded_label_territorio"] = df_clean["TerritorioCon"].apply(encode_target, args=[category_mapping_ter])
    df_clean["encoded_label_setor"] = df_clean['SetorInstitucionalCon'].apply(encode_target, args=[category_mapping_sec])

    return df_clean, inverted_mapping_ter, inverted_mapping_sec


def encode_target(label, category_mapping):
    """
    Encodes a label based on the given category mapping.
    If the label is not in the mapping, it assigns a new value and updates the mapping.

    Parameters:
        label (str): The label to encode.
        category_mapping (dict): A dictionary mapping labels to encoded values.

    Returns:
        int: The encoded value for the label.
    """
    if label not in category_mapping:
        new_value = len(category_mapping)
        category_mapping[label] = new_value
  
    return category_mapping[label]


def compute_avg_embedding(tokens, word2vec_model):
    """
    Computes the average embedding for a list of tokens using the provided Word2Vec model.

    Parameters:
        tokens (list): List of tokens.
        word2vec_model (Word2Vec): Word2Vec model containing word embeddings.

    Returns:
        numpy.ndarray: Average embedding for the tokens.
    """
    unknown_embedding=[0]*word2vec_model.vector_size
    embeddings = [word2vec_model.wv[token] for token in tokens if token in word2vec_model.wv]
    if embeddings:  # Embeddings found
        return np.array(embeddings).mean(axis=0)  # Return average embedding as a NumPy array
    else:  # No embeddings found
        return np.array(unknown_embedding)
    

def embbeded_model(df_clean):
    """
    Embeds the descriptions in the DataFrame using Word2Vec model.

    Parameters:
        df_clean (DataFrame): DataFrame containing the column 'DescricaoInstrumento'.

    Returns:
        DataFrame: DataFrame with an additional column 'tokenized_Descricao_text' containing tokenized descriptions.
        Word2Vec: Word2Vec model trained on the tokenized descriptions.
    """
    df_clean['tokenized_Descricao_text'] = df_clean['DescricaoInstrumento'].apply(lambda x: simple_preprocess(x))
    word2vec_model = Word2Vec(sentences=df_clean['tokenized_Descricao_text'], vector_size=100, window=5, min_count=1, workers=4)
    return df_clean, word2vec_model


def embedded_data(df_clean,  words2vec_model):
    """
    Computes average embeddings for tokenized descriptions and prepares data for modeling.

    Parameters:
        df_clean (DataFrame): DataFrame containing columns 'tokenized_Descricao_text', 'encoded_label_territorio', and 'encoded_label_setor'.
        word2vec_model (Word2Vec): Word2Vec model used to compute embeddings.

    Returns:
        DataFrame: Modified DataFrame with an additional column 'avg_embedding' containing the average embeddings.
        numpy.ndarray: Feature matrix X containing average embeddings.
        Series: Target variable y1 corresponding to 'encoded_label_territorio'.
        Series: Target variable y2 corresponding to 'encoded_label_setor'.
    """
    df_clean['avg_embedding'] = df_clean['tokenized_Descricao_text'].apply(compute_avg_embedding, args=(words2vec_model,))
    X = df_clean['avg_embedding'].apply(pd.Series).to_numpy()
    y1 = df_clean['encoded_label_territorio']
    y2 = df_clean['encoded_label_setor']

    return df_clean, X, y1, y2


def save_as_pickle(inverted_mapping_ter, inverted_mapping_sec, word2vec_model, clf_sec, clf_ter):
    """
    Saves the inverted mappings, Word2Vec model, and classifiers as pickle files.

    Parameters:
        inverted_mapping_ter (dict): Inverted mapping for 'TerritorioCon' column.
        inverted_mapping_sec (dict): Inverted mapping for 'SetorInstitucionalCon' column.
        word2vec_model (Word2Vec): Word2Vec model trained on descriptions.
        clf_sec: Classifier trained on 'SetorInstitucionalCon' column.
        clf_ter: Classifier trained on 'TerritorioCon' column.
    """
    with open('projetofinal/models/model_data.pkl', 'wb') as f:
        # Pickle the data objects (ensure they are picklable)
        pickle.dump(inverted_mapping_ter, f)
        pickle.dump(inverted_mapping_sec, f)
        pickle.dump(word2vec_model, f)
        pickle.dump(clf_sec, f)
        pickle.dump(clf_ter, f)


def read_pickle():
    """
    Reads the pickled model data from the file.

    Returns:
        dict: Inverted mapping for 'TerritorioCon' column.
        dict: Inverted mapping for 'SetorInstitucionalCon' column.
        Word2Vec: Word2Vec model trained on descriptions.
        clf_sec: Classifier trained on 'SetorInstitucionalCon' column.
        clf_ter: Classifier trained on 'TerritorioCon' column.
    """
    with open('projetofinal/models/model_data.pkl', 'rb') as f:
        # Load the data objects in the same order they were saved
        inverted_mapping_ter = pickle.load(f)
        inverted_mapping_sec = pickle.load(f)
        word2vec_model = pickle.load(f)
        clf_sec = pickle.load(f)
        clf_ter = pickle.load(f)

    return inverted_mapping_ter, inverted_mapping_sec, word2vec_model, clf_sec, clf_ter
        

def read_data(data_path, format):
    """
    Reads data from a file specified by the data path.

    Parameters:
        data_path (str): Path to the data file.
        format (str): Format of the data file. It can be 'xlsx' or any other format supported by pandas.

    Returns:
        DataFrame: DataFrame containing the data read from the file.
    """
    if format == "xlsx":
        df = pd.read_excel(data_path, sheet_name=2)
    else:
        df = pd.read_csv(data_path)

    return df