from projetofinal.preprocessing import (
    map_dict,
    embbeded_model,
    embedded_data,
    save_as_pickle,
    read_pickle,
    read_data,
)
from projetofinal.train_tools import train_split, pred_all, train_model, eval_model

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
    "--model_path",
    help="path where the model is in your computer and the name of your model",
    default="model_data.pkl",
)
@click.option(
    "--format",
    help="format of the file to read",
    default="xlsx",
    type=click.Choice(["xlsx", "csv"], case_sensitive=False),
)
def train(order, data_path, model_path, format):
    print(f"Paramns: Order - {order}, Data Path - {data_path}, Format - {format}")
    print("Reading Data")
    df = read_data(data_path, format)
    print("Data has been loaded")
    if order == "train":
        df_clean = df.copy()

        print("Preprocenssing Data")
        df_clean, inverted_mapping_ter, inverted_mapping_sec = map_dict(df_clean)
        print("Mapping calculated for territory and sector")
        print("Training word to vec model")
        df_clean, word2vec_model = embbeded_model(df_clean)
        print("Word Vec Model created")
        df_clean, X, y1, y2 = embedded_data(df_clean, word2vec_model)

        print("Training both models for territory and sector")
        clf_ter = train_model(X, y1)
        clf_sec = train_model(X, y2)
        # X_train_1, X_test_1, y_train_1, y_test_1, X_train_2, X_test_2, y_train_2, y_test_2 = train_split(X, y1, y2)

        print("Savind model")
        save_as_pickle(
            inverted_mapping_ter, inverted_mapping_sec, word2vec_model, clf_sec, clf_ter, model_path
        )
        print("Model Saved")

    elif order == "eval":
        print("Evaluating Model")
        print("Reading  saved models")
        (
            inverted_mapping_ter,
            inverted_mapping_sec,
            word2vec_model,
            clf_sec,
            clf_ter,
        ) = read_pickle(model_path)
        print("Model Loaded")
        print("Evaluating")
        df_eval = pred_all(
            df,
            word2vec_model,
            clf_ter,
            clf_sec,
            inverted_mapping_ter,
            inverted_mapping_sec,
        )
        print("print evaluations")
        eval_model(df_eval)

    else:
        print("Predictions")
        print("Reading the Saved Model")
        (
            inverted_mapping_ter,
            inverted_mapping_sec,
            word2vec_model,
            clf_sec,
            clf_ter,
        ) = read_pickle(model_path)
        print("Making the predictions")
        df_final = pred_all(
            df,
            word2vec_model,
            clf_ter,
            clf_sec,
            inverted_mapping_ter,
            inverted_mapping_sec,
        )
        print("Saving Final Dataset")
        df_final.to_csv("projetofinal/final_data/df_final.csv")


cli.add_command(train)

if __name__ == "__main__":
    cli()
    
