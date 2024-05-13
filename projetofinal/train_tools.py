from projetofinal.preprocessing import compute_avg_embedding
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import pandas as pd
from gensim.utils import simple_preprocess


def train_split(X, y1, y2):
    """
    Splits the data into training and testing sets for two different target variables.

    Parameters:
        X (numpy.ndarray): Feature matrix.
        y1 (array-like): Target variable 1.
        y2 (array-like): Target variable 2.

    Returns:
        tuple: X_train_1, X_test_1, y_train_1, y_test_1 for target variable 1.
        tuple: X_train_2, X_test_2, y_train_2, y_test_2 for target variable 2.
    """
    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(
        X, y1, test_size=0.1, random_state=42
    )
    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(
        X, y2, test_size=0.1, random_state=42
    )

    return (
        X_train_1,
        X_test_1,
        y_train_1,
        y_test_1,
        X_train_2,
        X_test_2,
        y_train_2,
        y_test_2,
    )


def train_model(X, y):
    """
    Trains a classification model using XGBoost.

    Parameters:
        X (numpy.ndarray): Feature matrix.
        y (array-like): Target variable.

    Returns:
        XGBClassifier: Trained XGBoost classifier.
    """
    clf = XGBClassifier(random_state=42, max_depth=5, verbose=1)
    clf.fit(X, y)
    return clf


def return_embeedings(
    string, word2vec_model, clf_t, clf_s, inverted_mapping_t, inverted_mapping_s
):
    """
    Returns predictions for the territory and sector based on the input string.

    Parameters:
        string (str): Input string to predict territory and sector.
        word2vec_model (Word2Vec): Word2Vec model used to compute embeddings.
        clf_t: Classifier for territory prediction.
        clf_s: Classifier for sector prediction.
        inverted_mapping_t (dict): Inverted mapping for territory prediction.
        inverted_mapping_s (dict): Inverted mapping for sector prediction.

    Returns:
        tuple: Predicted territory and sector as strings.
    """
    case = pd.DataFrame({"DescricaoInstrumento": [string]})
    case["tokenized_Descricao_text"] = case["DescricaoInstrumento"].apply(
        lambda x: simple_preprocess(x)
    )
    case["avg_embedding"] = case["tokenized_Descricao_text"].apply(
        compute_avg_embedding, args=(word2vec_model,)
    )

    X = case["avg_embedding"].apply(pd.Series).to_numpy()

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


def pred_all(
    df, word2vec_model, clf_ter, clf_sec, inverted_mapping_ter, inverted_mapping_sec
):
    """
    Predicts territories and sectors for all descriptions in the DataFrame.

    Parameters:
        df (DataFrame): DataFrame containing descriptions to predict territories and sectors for.
        word2vec_model (Word2Vec): Word2Vec model used to compute embeddings.
        clf_ter: Classifier for territory prediction.
        clf_sec: Classifier for sector prediction.
        inverted_mapping_ter (dict): Inverted mapping for territory prediction.
        inverted_mapping_sec (dict): Inverted mapping for sector prediction.

    Returns:
        DataFrame: DataFrame with predicted territories and sectors.
    """
    all_str_ter = []
    all_str_sec = []

    for i in range(len(df)):

        string = df.DescricaoInstrumento.iloc[i]
        str_pred_t, str_pred_s = return_embeedings(
            string,
            word2vec_model,
            clf_ter,
            clf_sec,
            inverted_mapping_ter,
            inverted_mapping_sec,
        )

        all_str_ter.append(str_pred_t)
        all_str_sec.append(str_pred_s)

    df["sec_pred"] = all_str_sec
    df["ter_pred"] = all_str_ter

    return df


def eval_model(df):
    """
    Evaluates the accuracy of territory and sector predictions.

    Parameters:
        df (DataFrame): DataFrame containing true and predicted territory and sector labels.

    Prints:
        float: Accuracy of territory predictions.
        float: Accuracy of sector predictions.
    """
    y_sec_pred = df["sec_pred"]
    y_ter_pred = df["ter_pred"]
    y_sec_true = df["SetorInstitucionalCon"]
    y_ter_true = df["TerritorioCon"]

    accuracy_sec = accuracy_score(y_sec_true, y_sec_pred)
    accuracy_ter = accuracy_score(y_ter_true, y_ter_pred)
    print("Accuracy Territory:", accuracy_ter)
    print("Accuracy Sector:", accuracy_sec)
