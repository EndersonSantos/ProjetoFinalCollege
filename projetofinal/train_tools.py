from projetofinal.preprocessing import compute_avg_embedding

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import csv
import os
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


def train_model(X, y, model_name):
    """
    Trains a classification model using XGBoost.

    Parameters:
        X (numpy.ndarray): Feature matrix.
        y (array-like): Target variable.

    Returns:
        XGBClassifier: Trained XGBoost classifier.
    """
    if model_name=="xg_boost":
        model = XGBClassifier(random_state=42, max_depth=5, verbose=1)
    elif model_name=="decision_tree":
        model = DecisionTreeClassifier(max_depth=3)
    elif model_name=="knn":
        model = KNeighborsClassifier(n_neighbors=5)
    elif model_name=="logistic":
        model = LogisticRegression(multi_class='multinomial')
    elif model_name=="svm":
        model = SVC()
    model.fit(X, y)
    return model


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
    prediction_t_proba = clf_t.predict_proba(X).max()
    str_pred_t = map_numbers_to_categories(prediction_t, inverted_mapping_t)

    prediction_s = clf_s.predict(X)
    prediction_s_proba = clf_s.predict_proba(X).max()
    str_pred_s = map_numbers_to_categories(prediction_s, inverted_mapping_s)

    return str_pred_t[0], str_pred_s[0], prediction_t_proba, prediction_s_proba


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
    all_prob_ter = []
    all_prob_sec = []

    for i in range(len(df)):

        string = df.DescricaoInstrumento.iloc[i]
        str_pred_t, str_pred_s, prediction_t_proba, prediction_s_proba = (
            return_embeedings(
                string,
                word2vec_model,
                clf_ter,
                clf_sec,
                inverted_mapping_ter,
                inverted_mapping_sec,
            )
        )

        all_str_ter.append(str_pred_t)
        all_str_sec.append(str_pred_s)
        all_prob_ter.append(round(prediction_t_proba, 3))
        all_prob_sec.append(round(prediction_s_proba, 3))

    df["sec_pred"] = all_str_sec
    df["ter_pred"] = all_str_ter
    df["sec_probabilidade"] = all_prob_ter
    df["ter_probabilidade"] = all_prob_sec

    return df


def eval_model(df, model_name):
    """
    Evaluate a model's performance based on the given dataframe and return the results
    as a new row for logging or further analysis.

    The function calculates accuracy, precision, and recall for both territory and sector 
    predictions. The results are rounded to two decimal places and printed out. The function 
    then returns these metrics along with the model name as a list.

    Args:
        df (pandas.DataFrame): DataFrame containing the true and predicted labels.
            Expected columns are 'sec_pred', 'ter_pred', 'SetorInstitucionalCon', and 'TerritorioCon'.
        model_name (str): The name of the model being evaluated.

    Returns:
        list: A list containing the model name, accuracy, precision, and recall metrics for both
              territory and sector predictions.

    Example:
        new_row = eval_model(df, "MyModel")
    """
    y_sec_pred = df["sec_pred"]
    y_ter_pred = df["ter_pred"]
    y_sec_true = df["SetorInstitucionalCon"]
    y_ter_true = df["TerritorioCon"]

    accuracy_ter = np.round(accuracy_score(y_ter_true, y_ter_pred),2)
    precision_ter = np.round(precision_score(y_ter_true, y_ter_pred, average='weighted'),2)
    recall_ter = np.round(recall_score(y_ter_true, y_ter_pred, average='weighted'),2)
    
    accuracy_sec = np.round(accuracy_score(y_sec_true, y_sec_pred),2)
    precision_sec = np.round(precision_score(y_sec_true, y_sec_pred, average='weighted'),2)
    recall_sec = np.round(recall_score(y_sec_true, y_sec_pred, average='weighted'),2)

    print("Accuracy Territory:", accuracy_ter)
    print("Precision Territory:", precision_ter)
    print("Recall Territory:", recall_ter)
    print("Accuracy Sector:", accuracy_sec)
    print("Precision Sector:", precision_sec)
    print("Recall Sector:", recall_sec)

    new_row = [model_name, accuracy_ter, precision_ter, recall_ter, accuracy_sec, precision_sec, recall_sec]

    return new_row


def save_new_row(new_row):
    
    """
    Append a new row to the 'performance.csv' file. If the file does not exist, create it 
    and add a header row before appending the new row.

    Args:
        new_row (list): The row to be added to the CSV file. It should contain the values in the 
                        order that matches the header columns.

    Example:
        save_new_row([1, 2, 3, 4, 5, 6])
    """
    csv_file_path = 'performance.csv'
    file_exists = os.path.exists(csv_file_path)
    
    with open("performance.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        if not file_exists:
            header =  ["Model Name", "Accuracy Territory", "Precision Territory", "Recall Territory", "Accuracy Sector", "Precision Sector", "Recall Sector"]
            writer.writerow(header)

        # Write the new row
        writer.writerow(new_row)