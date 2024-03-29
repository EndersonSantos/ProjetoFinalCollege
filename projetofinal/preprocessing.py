from gensim.utils import simple_preprocess
from gensim.models import Word2Vec
import numpy as np
import pandas  as pd
import pickle


def map_dict(df_clean):
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
  if label not in category_mapping:
    new_value = len(category_mapping)
    category_mapping[label] = new_value
  
  return category_mapping[label]


def compute_avg_embedding(tokens, word2vec_model):
    unknown_embedding=[0]*word2vec_model.vector_size
    embeddings = [word2vec_model.wv[token] for token in tokens if token in word2vec_model.wv]
    if embeddings:  # Embeddings found
        return np.array(embeddings).mean(axis=0)  # Return average embedding as a NumPy array
    else:  # No embeddings found
        return np.array(unknown_embedding)
    

def embbeded_model(df_clean):
    df_clean['tokenized_Descricao_text'] = df_clean['DescricaoInstrumento'].apply(lambda x: simple_preprocess(x))
    word2vec_model = Word2Vec(sentences=df_clean['tokenized_Descricao_text'], vector_size=100, window=5, min_count=1, workers=4)
    return df_clean, word2vec_model


def embedded_data(df_clean,  words2vec_model):

    df_clean['avg_embedding'] = df_clean['tokenized_Descricao_text'].apply(compute_avg_embedding, args=(words2vec_model,))
    X = df_clean['avg_embedding'].apply(pd.Series).to_numpy()
    y1 = df_clean['encoded_label_territorio']
    y2 = df_clean['encoded_label_setor']

    return df_clean, X, y1, y2


def save_as_pickle(inverted_mapping_ter, inverted_mapping_sec, word2vec_model, clf_sec, clf_ter):
    with open('projetofinal/models/model_data.pkl', 'wb') as f:
        # Pickle the data objects (ensure they are picklable)
        pickle.dump(inverted_mapping_ter, f)
        pickle.dump(inverted_mapping_sec, f)
        pickle.dump(word2vec_model, f)
        pickle.dump(clf_sec, f)
        pickle.dump(clf_ter, f)


def read_pickle():
    with open('projetofinal/models/model_data.pkl', 'rb') as f:
        # Load the data objects in the same order they were saved
        inverted_mapping_ter = pickle.load(f)
        inverted_mapping_sec = pickle.load(f)
        word2vec_model = pickle.load(f)
        clf_sec = pickle.load(f)
        clf_ter = pickle.load(f)

    return inverted_mapping_ter, inverted_mapping_sec, word2vec_model, clf_sec, clf_ter
        