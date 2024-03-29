from projetofinal.preprocessing import map_dict, embbeded_model, embedded_data, save_as_pickle, read_pickle
from projetofinal.train_tools import train_split, pred_all, train_model

import click
import pandas as pd
import pickle
import os


@click.group()
def cli():
    # At the moment all options belong to train command so this is empty
    pass


@click.command()
@click.option(
    "--order",
    help="",
    default="train",
    type=click.Choice(["train", "eval", "pred", "app"], case_sensitive=False),
)
@click.option(
    "--data_path",
    help="Name of the dataset path",
    default="projetofinal/data_train/01.Dataset FI_06032024.xlsx",
)
@click.option(
    "--format",
    help="format of the file to read",
    default="xlsx",
)
def train(
    order,
    data_path,
    format
):
    
    if  order == "train":
        current_path = os.getcwd()
        # Print the path
        print(f"Current working directory: {current_path}")
        df = pd.read_excel(data_path, sheet_name=2)
        df_clean = df.copy()

        df_clean, inverted_mapping_ter, inverted_mapping_sec = map_dict(df_clean)
        df_clean, word2vec_model = embbeded_model(df_clean)
        
        df_clean, X, y1, y2 = embedded_data(df_clean,  word2vec_model)

        clf_ter = train_model(X, y1)
        clf_sec = train_model(X, y2)
        #X_train_1, X_test_1, y_train_1, y_test_1, X_train_2, X_test_2, y_train_2, y_test_2 = train_split(X, y1, y2)

        save_as_pickle(inverted_mapping_ter, inverted_mapping_sec, word2vec_model, clf_sec, clf_ter)
    else:
        inverted_mapping_ter, inverted_mapping_sec, word2vec_model, clf_sec, clf_ter = read_pickle()

        if format == 'xlsx':
            df = pd.read_excel(data_path)
        elif  format == 'csv':
            df = pd.read_csv(data_path)

        df_final = pred_all(df, word2vec_model, clf_ter, clf_sec, inverted_mapping_ter, inverted_mapping_sec)
        df_final.to_csv("projetofinal/final_data/df_final.csv")

cli.add_command(train)

if __name__ == "__main__":
    cli()