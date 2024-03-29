from projetofinal.preprocessing import compute_avg_embedding

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pandas as pd
from gensim.utils import simple_preprocess


def train_split(X, y1, y2):

    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X, y1, test_size=0.1, random_state=42)
    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X, y2, test_size=0.1, random_state=42)

    return X_train_1, X_test_1, y_train_1, y_test_1, X_train_2, X_test_2, y_train_2, y_test_2


def train_model(X, y):
    clf = XGBClassifier(random_state=42, max_depth=5)
    clf.fit(X, y)
    return clf


def return_embeedings(string, word2vec_model, clf_t, clf_s, inverted_mapping_t, inverted_mapping_s):
    case = pd.DataFrame({'DescricaoInstrumento': [string]})
    case['tokenized_Descricao_text'] = case['DescricaoInstrumento'].apply(lambda x: simple_preprocess(x))
    case['avg_embedding'] = case['tokenized_Descricao_text'].apply(compute_avg_embedding, args=(word2vec_model,))
    
    X = case['avg_embedding'].apply(pd.Series).to_numpy()
    
    prediction_t = clf_t.predict(X)
    str_pred_t = map_numbers_to_categories(prediction_t, inverted_mapping_t)
    
    prediction_s = clf_s.predict(X)
    str_pred_s = map_numbers_to_categories(prediction_s, inverted_mapping_s)
    
    return str_pred_t[0], str_pred_s[0]


def map_numbers_to_categories(numbers, category_mapping):
    """Maps numbers back to their corresponding category names using a provided mapping dictionary.

    Args:
        numbers: A list or array containing the numerical representations of categories.
        category_mapping: A dictionary mapping category names (keys) to their numerical representations (values).

    Returns:
        A list containing the corresponding category names for the input numbers.
    """

    category_names = [category_mapping.get(number, None) for number in numbers]
    return category_names


def pred_all(df, word2vec_model, clf_ter, clf_sec, inverted_mapping_ter, inverted_mapping_sec):
    all_str_ter = []
    all_str_sec = []

    for i in range(len(df)):

        string = df.DescricaoInstrumento.iloc[i]
        str_pred_t, str_pred_s = return_embeedings(string, word2vec_model, clf_ter, clf_sec, inverted_mapping_ter, inverted_mapping_sec)
        
        all_str_ter.append(str_pred_t)
        all_str_sec.append(str_pred_s)
        
    df["sec_pred"] = all_str_sec  
    df["ter_pred"] = all_str_ter

    return df