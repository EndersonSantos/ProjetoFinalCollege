import pandas as pd
import streamlit as st
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import pickle
from projetofinal.preprocessing import compute_avg_embedding, read_pickle
from projetofinal.train_tools import return_embeedings
import subprocess
import os
import pydeck as pdk
from urllib.error import URLError
import geopandas as gpd
import folium
from streamlit_folium import folium_static
import ipywidgets as widgets
from IPython.display import display, clear_output
import datetime
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report  
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# Apply the theme
st.set_page_config(
    page_title="Decoding Movements Dashboard",
    page_icon=":chart_with_upwards_trend:",
    menu_items={
        "About": "This app predicts the sector and territory of a financial instrument based on its description."
    },
)

# Use markdown for a custom-styled title and subtitle
st.markdown(
    "<h1 style='text-align: center;'>Decoding Movements</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h4 style='text-align: center; color: gray;'>Prever, através de descritivos, os setor e território de uma transação de um fundo de investimento.</h4>",
    unsafe_allow_html=True,
)

def try_read_csv(uploaded_file):
    encodings = ["utf-8", "iso-8859-1", "cp1252"]
    for enc in encodings:
        try:
            return pd.read_csv(uploaded_file, encoding=enc, delimiter=";"), enc
        except UnicodeDecodeError:
            continue
    return None, None


def home():
    # Model Selection Dropdown
    # Assuming your models are stored with recognizable names in 'projetofinal/models'
    model_files = os.listdir('projetofinal/models')
    model_names = [file.replace('.pkl', '') for file in model_files if file.endswith('.pkl')]
    
    default_model_index = model_names.index('xg_boost') if 'xg_boost' in model_names else 0

    model_name = st.selectbox("Choose a model to use:", options=model_names, index=default_model_index)
    # Initialize session state
    if "selected_example" not in st.session_state:
        st.session_state["selected_example"] = ""

    # Text input for description
    st.subheader("Description")
    user_input = st.text_input("", value=st.session_state["selected_example"], placeholder='Enter a Description')

    # Load pre-trained models and other necessary components based on the selected model
    if model_name:
        model_path = f'projetofinal/models/{model_name}.pkl'
        inverted_mapping_ter, inverted_mapping_sec, word2vec_model, clf_sec, clf_ter = read_pickle(model_path)

    # Predict button
    if st.button("Predict"):
        if user_input and model_name:
            tokens = simple_preprocess(user_input)
            avg_embedding = compute_avg_embedding(tokens, word2vec_model)
            pred_territory, pred_sector, prediction_t_proba, prediction_s_proba = return_embeedings(
                user_input,
                word2vec_model,
                clf_ter,
                clf_sec,
                inverted_mapping_ter,
                inverted_mapping_sec,
                model_name
            )
            st.success(f"Predicted Territory: {pred_territory}")
            st.success(f"Predicted Sector: {pred_sector}")
        else:
            st.error("Please enter a description and select a model.")



    # Quick example buttons
    st.write("Quick Examples:")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("KFW 2.875% 29/05/26"):
            st.session_state["selected_example"] = "KFW 2.875% 29/05/26"
    with col2:
        if st.button("HEIBOS 0.25 13/10/24"):
            st.session_state["selected_example"] = "HEIBOS 0.25 13/10/24"
    with col3:
        if st.button("FIS 0.125 03/12/22"):
            st.session_state["selected_example"] = "FIS 0.125 03/12/22"

    # File upload for prediction
    st.header("Upload a File")
    uploaded_file = st.file_uploader(
        "Upload your file (CSV or Excel)", type=["csv", "xlsx"], key="file-uploader"
    )


    # Function to process and predict the uploaded file
    def process_and_predict_file(df, model_name):
        # Load model components based on selected model_name
        model_path = f'projetofinal/models/{model_name}.pkl'
        inverted_mapping_ter, inverted_mapping_sec, word2vec_model, clf_sec, clf_ter = read_pickle(model_path)
        
        # Initialize columns for predictions
        df["Predicted Territory"] = ""
        df["Predicted Sector"] = ""

        # Preprocess and predict for each row
        for index, row in df.iterrows():
            description = row["DescricaoInstrumento"]
            tokens = description.lower().split()  # Example tokenization, adjust as needed
            avg_embedding = compute_avg_embedding(tokens, word2vec_model)

            # Making predictions
            pred_territory, pred_sector, prediction_t_proba, prediction_s_proba = return_embeedings(
                description,
                word2vec_model,
                clf_ter,
                clf_sec,
                inverted_mapping_ter,
                inverted_mapping_sec,
                model_name
            )

            # Assigning predictions to the dataframe
            df.at[index, "Predicted Territory"] = pred_territory
            df.at[index, "Predicted Sector"] = pred_sector

        return df

    # Temporary store the uploaded file and show preview
    if uploaded_file is not None:
        if uploaded_file.type == "text/csv":
            df, encoding_used = try_read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine="openpyxl")

        required_columns = ["DescricaoInstrumento"]

        # Check if the uploaded file contains all required columns
        if not all(column in df.columns for column in required_columns):
            st.error(f"The uploaded file does not contain all required columns: {', '.join(required_columns)}")
        else:
            # Store the DataFrame in session state to use later
            st.session_state['uploaded_df'] = df
            
            # Display the first 10 rows of the DataFrame
            st.write("Preview of uploaded file (first 10 rows):")
            st.dataframe(df.head(10))


    # Button to trigger prediction
    if 'uploaded_df' in st.session_state and st.button("Predict from File"):
        processed_df = process_and_predict_file(st.session_state['uploaded_df'], model_name)
        st.session_state['processed_df'] = processed_df  # Store processed data for potential download
        st.write("Predictions completed. Review the predicted data below:")
        st.dataframe(processed_df.head(10))  # Show first 10 rows of the processed data

        # Show download button after predictions
        csv = processed_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download predictions as CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )


def analysis():
    st.title("Análises")
    st.write("Nesta página, é possível observar algumas análises sobre o Território e o Setor. No Território, apresentamos os cinco principais registos de cada país, enquanto no Setor disponibilizamos uma tabela com os registos mais frequentes. Tudo isto é realizado através do token.")

    st.markdown("## Análise dos Modelos")

    # Load performance data
    performance_df = pd.read_csv('performance.csv')

    # Function to suggest the best models based on the average of selected metrics
    def suggest_best_models(performance_df, metric_columns):
        # Calculate the average of selected metrics
        performance_df['Aggregate Score'] = performance_df[metric_columns].mean(axis=1)
        # Sort models by average in descending order and get the top 3
        best_models = performance_df.nlargest(3, 'Aggregate Score')['Model Name'].tolist()
        return best_models

    # Create a metric selector
    selected_metric = st.selectbox("Select the metric:", options=['Accuracy', 'Precision', 'Recall'])

    # Define metric columns based on selected metric
    if selected_metric == 'Accuracy':
        metric_columns = ['Accuracy Territory', 'Accuracy Sector']
        metric_name = 'Accuracy'
    elif selected_metric == 'Precision':
        metric_columns = ['Precision Territory', 'Precision Sector']
        metric_name = 'Precision'
    elif selected_metric == 'Recall':
        metric_columns = ['Recall Territory', 'Recall Sector']
        metric_name = 'Recall'

    # Prepare data for the plot
    data = []
    for column in metric_columns:
        data.append(go.Bar(x=performance_df['Model Name'], y=performance_df[column], name=column))

    # Create the figure
    fig = go.Figure(data)

    # Update layout
    fig.update_layout(barmode='group',
                    title=f'{selected_metric} per Model',
                    xaxis_title='Model',
                    yaxis_title=f'{selected_metric}')

    # Display the plot
    st.plotly_chart(fig)

    # Suggest the best models
    best_models = suggest_best_models(performance_df, metric_columns)
    st.write(f"Os melhores modelos com base na média entre {metric_name} Territory e {metric_name} Sector são: {', '.join(best_models)}")


    # Load territorial data
    territory_df = pd.read_csv("projetofinal/analysis/territorio_df.csv")

    # Load a world shapefile
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    merged = world.merge(territory_df, how='left', left_on='iso_a3', right_on='Território')

    # Create a folium map centered at an initial location
    m = folium.Map(location=[0, 0], zoom_start=2)

    # Function to handle click on a country
    def click_handler(feature, **kwargs):
        iso_a3 = feature['iso_a3']
        top5 = territory_df[territory_df['Território'] == iso_a3].nlargest(5, 'Contagem')
        popup_text = f"{feature['Descrição do território']} Top 5:\n"
        if not top5.empty:
            for _, row in top5.iterrows():
                popup_text += f"{row['Token']}:{row['Contagem']}\n"
        return popup_text

    # Add GeoDataFrame to the folium map
    folium.GeoJson(
        merged,
        name='geojson',
        tooltip=folium.GeoJsonTooltip(fields=['Descrição do território']),
        style_function=lambda feature: {'fillColor': '#0E7A0D' if feature['properties']['Território'] in territory_df['Território'].values else '#ffffff', 'color': '#003300'},  # fill color for countries with tokens in top 5
        highlight_function=lambda feature: {'weight': 0},  # remove country highlights
        popup=folium.Popup(lambda feature: click_handler(feature, territory_df), max_width=400)  # increase max width of popup text box
    ).add_to(m)

    # Add click event handlers for each country
    for _, feature in merged.iterrows():
        iso_a3 = feature['iso_a3']
        top5 = territory_df[territory_df['Território'] == iso_a3].nlargest(5, 'Contagem')
        if not top5.empty:
            geojson = folium.GeoJson(
                feature['geometry'],
                style_function=lambda feature: {'fillColor': '#0E7A0D', 'color': '#003300'},  # fill color for countries with tokens in top 5
                highlight_function=lambda feature: {'weight': 0},  # remove country highlights
                name='geojson',
                popup=folium.Popup(click_handler(feature), max_width=400)  # increase max width of popup text box
            )
        else:
            geojson = folium.GeoJson(
                feature['geometry'],
                style_function=lambda feature: {'fillColor': '#ffffff', 'color': '#000000'},  # remove fill color for countries without list
                highlight_function=lambda feature: {'weight': 0},  # remove country highlights
                name='geojson',
                popup=None  # no text box for countries without list
            )
        geojson.add_to(m)

    # Display the interactive map
    st.markdown("### Mapa - Análise de tokens pelo Território")
    folium_static(m)

    # Load sector data
    sector_df = pd.read_csv('projetofinal/analysis/setor_df.csv')
    sector_df1 = sector_df.drop(['Unnamed: 0','Descrição do Setor Institucional'], axis = 1)

    # Function to display all records for a given sector
    def records_per_sector(sector):
        return sector_df1[sector_df['Setor Institucional'] == sector]

    # Function to update the list of records when a new sector is selected
    def update_records(selected_sector):
        st.write(f"Resgitos do {selected_sector} ({sector_df.loc[sector_df['Setor Institucional'] == selected_sector, 'Descrição do Setor Institucional'].iloc[0]}):")
        records = records_per_sector(selected_sector)
        st.write(records)

    # Get the unique list of institutional sectors
    st.markdown("### Análise de Tokens por Sector")
    sectors = sector_df1['Setor Institucional'].unique()

    # Create the dropdown menu to select the sector
    selected_sector = st.selectbox("Selecione o sector:", sectors, index=0)

    # Define the callback function to update records when a new sector is selected
    update_records(selected_sector)



def encode_target(label, category_mapping):
    if label not in category_mapping:
        new_value = len(category_mapping)
        category_mapping[label] = new_value
    return category_mapping[label]

def map_numbers_to_categories(numbers, category_mapping):
    return [category_mapping.get(number, None) for number in numbers]

def compute_avg_embedding(tokens, word2vec_model, unknown_embedding=[0]*100):
    embeddings = [word2vec_model.wv[token] for token in tokens if token in word2vec_model.wv]
    if embeddings:
        return np.array(embeddings).mean(axis=0)
    else:
        return np.array(unknown_embedding)

def training():
    st.title("Training Models")
    st.write("Here you can initiate training of models based on the latest data.")

    # Model Selection Dropdown
    model_files = os.listdir('projetofinal/models')
    model_names = [file.replace('.pkl', '') for file in model_files if file.endswith('.pkl')]
    
    default_model_index = model_names.index('xg_boost') if 'xg_boost' in model_names else 0
    model_name = st.selectbox("Choose a model to retrain:", options=model_names, index=default_model_index)

    # File upload for bulk data
    st.header("Upload a File for Retraining")
    bulk_uploaded_file = st.file_uploader(
        "Upload your file (CSV or Excel) containing multiple instances", type=["csv", "xlsx"], key="bulk-file-uploader"
    )

    required_columns = [
        "DescricaoInstrumento", "SetorInstitucionalCon", "TerritorioCon"
    ]

    # Process bulk upload
    if bulk_uploaded_file is not None:
        if bulk_uploaded_file.type == "text/csv":
            bulk_df, encoding_used = try_read_csv(bulk_uploaded_file)
        else:
            bulk_df = pd.read_excel(bulk_uploaded_file, engine="openpyxl")

        # Check if the uploaded file contains all required columns
        if not all(column in bulk_df.columns for column in required_columns):
            st.error(f"The uploaded file does not contain all required columns: {', '.join(required_columns)}")
        else:
            # Display the first 10 rows of the DataFrame
            st.write("Preview of uploaded file (first 10 rows):")
            st.dataframe(bulk_df.head(10))

            # Append bulk data to existing data and proceed with retraining
            st.session_state['bulk_uploaded_df'] = bulk_df


    
    st.header("Add a single instance")
    with st.form(key='training_form'):
        CodEntidadeRef = st.text_input("CodEntidadeRef")
        TipoInformacao = st.selectbox("TipoInformacao", ["A", "P"])
        TipoInstrumento = st.selectbox("TipoInstrumento", [
            "F21", "F22", "F29", "F3_P", "F4", "F511", "F512", "F519", "F521", "F522", "F71"
        ])
        DescricaoInstrumento = st.text_input("DescricaoInstrumento")
        MaturidadeOriginal = st.selectbox("MaturidadeOriginal", [
            "01", "10", "06", "07", "08", "_Z"
        ])
        SetorInstitucionalCon = st.selectbox("SetorInstitucionalCon", [
            "S11", "S121", "S122", "S123", "S124", "S125", "S126", "S127", "S128", "S129", 
            "S1311", "S1312", "S1313", "S1314", "S14", "S15"
        ])
        TerritorioCon = st.selectbox("TerritorioCon", [
            "_Z", "1E", "4C", "4D", "4S", "4W", "5C", "5D", "5F", "5H", "5U", "5X", 
            "7E", "7M", "ABW", "AFG", "AGO", "AND", "ARE", "ARG", "ARM", "ATG", 
            "AUS", "AUT", "AZE", "BEL", "BEN", "BGD", "BGR", "BHR", "BHS", "BIH", 
            "BLR", "BLZ", "BMU", "BOL", "BRA", "BRB", "BWA", "CAF", "CAN", "CHE", 
            "CHL", "CHN", "CIV", "CMR", "COD", "COG", "COL", "CPV", "CRI", "CUB", 
            "CUW", "CYM", "CYP", "CZE", "DEU", "DJI", "DNK", "DOM", "DZA", "ECU", 
            "EGY", "ESP", "EST", "ETH", "FIN", "FLK", "FRA", "FRO", "FSM", "GAB", 
            "GBR", "GEO", "GGY", "GHA", "GIB", "GIN", "GLP", "GNB", "GNQ", "GRC", 
            "HKG", "HRV", "HUN", "IDN", "IMN", "IND", "IRL", "IRN", "IRQ", "ISL", 
            "ISR", "ITA", "JEY", "JOR", "JPN", "KAZ", "KEN", "KGZ", "KHM", "KNA", 
            "KOR", "KWT", "LAO", "LBN", "LBR", "LIE", "LKA", "LTU", "LUX", "LVA", 
            "MAC", "MAR", "MCO", "MDA", "MDV", "MEX", "MKD", "MLI", "MLT", "MNE", 
            "MNG", "MOZ", "MRT", "MTQ", "MUS", "MWI", "MYS", "NAM", "NER", "NGA", 
            "NLD", "NOR", "NZL", "OMN", "PAK", "PAN", "PER", "PHL", "POL", "PRI", 
            "PRT", "PRY", "PSE", "QAT", "REU", "ROU", "RUS", "SAU", "SDN", "SEN", 
            "SGP", "SLV", "SRB", "STP", "SVK", "SVN", "SWE", "SWZ", "SYC", "TCA", 
            "TGO", "THA", "TLS", "TTO", "TUN", "TUR", "TWN", "TZA", "UGA", "UKR", 
            "URY", "USA", "UZB", "VCT", "VEN", "VGB", "VNM", "ZAF", "ZMB", "ZWE"
        ])

        submit_button = st.form_submit_button(label='Add and Retrain')

    if submit_button or 'bulk_uploaded_df' in st.session_state:
        if 'bulk_uploaded_df' in st.session_state:
            new_data_df = st.session_state['bulk_uploaded_df']
        else:
            new_data = {
                "CodEntidadeRef": CodEntidadeRef,
                "TipoInformacao": TipoInformacao,
                "TipoInstrumento": TipoInstrumento,
                "DescricaoInstrumento": DescricaoInstrumento,
                "MaturidadeOriginal": MaturidadeOriginal,
                "SetorInstitucionalCon": SetorInstitucionalCon,
                "TerritorioCon": TerritorioCon
            }
            new_data_df = pd.DataFrame([new_data])

        # Add missing columns with NA values
        missing_columns = ['CodEntidadeCon']
        for col in missing_columns:
            new_data_df[col] = pd.NA

        # Load existing training data
        training_data_path = "projetofinal/data_train/data_train.csv" 
        existing_data = pd.read_csv(training_data_path)

        # Append new data to existing data
        updated_data = pd.concat([existing_data, new_data_df], ignore_index=True)

        # Save updated data and backup old data
        current_datetime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        backup_data_path = f"projetofinal/data_train/backup/{current_datetime}_data_train.csv"
        updated_data_path = "projetofinal/data_train/data_train.csv"
        existing_data.to_csv(backup_data_path, index=False)  
        updated_data.to_csv(updated_data_path, index=False)  

        st.write("New data added successfully. Retraining the model...")

        # Preprocessing
        updated_data['tokenized_Descricao_text'] = updated_data['DescricaoInstrumento'].apply(lambda x: simple_preprocess(x))
        word2vec_model = Word2Vec(sentences=updated_data['tokenized_Descricao_text'], vector_size=100, window=5, min_count=1, workers=4)

        updated_data['avg_embedding'] = updated_data['tokenized_Descricao_text'].apply(lambda tokens: compute_avg_embedding(tokens, word2vec_model))
        X = updated_data['avg_embedding'].apply(pd.Series).to_numpy()

        unique_categories_ter = updated_data['TerritorioCon'].unique()
        category_mapping_ter = dict(zip(unique_categories_ter, range(len(unique_categories_ter))))
        inverted_mapping_ter = {value: key for key, value in category_mapping_ter.items()}

        unique_categories_sec = updated_data["SetorInstitucionalCon"].unique()
        category_mapping_sec = dict(zip(unique_categories_sec, range(len(unique_categories_sec))))
        inverted_mapping_sec = {value: key for key, value in category_mapping_sec.items()}

        updated_data["encoded_label_territorio"] = updated_data["TerritorioCon"].apply(encode_target, args=[category_mapping_ter])
        updated_data["encoded_label_setor"] = updated_data['SetorInstitucionalCon'].apply(encode_target, args=[category_mapping_sec])

        y1 = updated_data['encoded_label_territorio']
        y2 = updated_data['encoded_label_setor']

        # Train Test Split
        X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X, y1, test_size=0.1, random_state=41)
        X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X, y2, test_size=0.1, random_state=41)

        # Model selection and training
        if model_name == "xg_boost":
            model_ter = XGBClassifier(random_state=42, max_depth=5)
            model_sec = XGBClassifier(random_state=42, max_depth=5)
        elif model_name == "decision_tree":
            model_ter = DecisionTreeClassifier(max_depth=3)
            model_sec = DecisionTreeClassifier(max_depth=3)
        elif model_name == "knn":
            model_ter = KNeighborsClassifier(n_neighbors=5)
            model_sec = KNeighborsClassifier(n_neighbors=5)
        elif model_name == "logistic":
            model_ter = LogisticRegression(multi_class='multinomial')
            model_sec = LogisticRegression(multi_class='multinomial')
        elif model_name == "svm":
            model_ter = SVC()
            model_sec = SVC()

        model_ter.fit(X_train_1, y_train_1)
        model_sec.fit(X_train_2, y_train_2)

        # Backup old models
        backup_path = f"projetofinal/models/backup/{current_datetime}_{model_name}.pkl"
        os.rename(f'projetofinal/models/{model_name}.pkl', backup_path)

        # Save the retrained models
        model_path = f'projetofinal/models/{model_name}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(inverted_mapping_ter, f)
            pickle.dump(inverted_mapping_sec, f)
            pickle.dump(word2vec_model, f)
            pickle.dump(model_sec, f)
            pickle.dump(model_ter, f)

        st.write(f"Model {model_name} retrained and saved successfully.")

        # Model evaluation
        st.write("Evaluating the retrained model...")

        # Predictions on the test set
        y_pred_ter = model_ter.predict(X_test_1)
        y_pred_sec = model_sec.predict(X_test_2)

        # Metrics calculation
        accuracy_ter = accuracy_score(y_test_1, y_pred_ter)
        precision_ter = precision_score(y_test_1, y_pred_ter, average='weighted')
        recall_ter = recall_score(y_test_1, y_pred_ter, average='weighted')

        accuracy_sec = accuracy_score(y_test_2, y_pred_sec)
        precision_sec = precision_score(y_test_2, y_pred_sec, average='weighted')
        recall_sec = recall_score(y_test_2, y_pred_sec, average='weighted')

        # Display metrics
        st.write("### Territory Model Performance")
        st.write(f"Accuracy: {accuracy_ter:.4f}")
        st.write(f"Precision: {precision_ter:.4f}")
        st.write(f"Recall: {recall_ter:.4f}")

        st.write("### Sector Model Performance")
        st.write(f"Accuracy: {accuracy_sec:.4f}")
        st.write(f"Precision: {precision_sec:.4f}")
        st.write(f"Recall: {recall_sec:.4f}")

        # Classification report
        st.write("### Territory Model Classification Report")
        st.text(classification_report(y_test_1, y_pred_ter))

        st.write("### Sector Model Classification Report")
        st.text(classification_report(y_test_2, y_pred_sec))

        # Confusion matrices
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        sns.heatmap(pd.crosstab(y_test_1, y_pred_ter), annot=True, fmt='d', ax=axes[0])
        axes[0].set_title('Territory Model Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')

        sns.heatmap(pd.crosstab(y_test_2, y_pred_sec), annot=True, fmt='d', ax=axes[1])
        axes[1].set_title('Sector Model Confusion Matrix')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')

        st.pyplot(fig)

        st.write("Retraining and evaluation completed.")

        # Update performance.csv
        performance_df = pd.read_csv("performance.csv")
        new_performance_row = {
            'Model Name': model_name,
            'Accuracy Territory': round(accuracy_ter, 2),
            'Precision Territory': round(precision_ter, 2),
            'Recall Territory': round(recall_ter, 2),
            'Accuracy Sector': round(accuracy_sec, 2),
            'Precision Sector': round(precision_sec, 2),
            'Recall Sector': round(recall_sec, 2)
        }

        # Check if the model already exists in the dataframe
        if model_name in performance_df['Model Name'].values:
            performance_df.update(performance_df.loc[performance_df['Model Name'] == model_name, new_performance_row.keys()].assign(**new_performance_row))
        else:
            performance_df = performance_df.append(new_performance_row, ignore_index=True)

        performance_df.to_csv("performance.csv", index=False)
        st.write("Model performance updated in performance.csv.")

        st.write("Retraining and evaluation completed.")


pages = {
    "Principal": {"function": home, "icon": "🏠"},
    "Análises": {"function": analysis, "icon": "📊"},
    "Training": {"function": training, "icon": "🏋️"}
}

selection = st.sidebar.radio("Go to", [f"{pages[key]['icon']} {key}" for key in pages.keys()], index=0)

page_name = selection.split()[1]  
pages[page_name]["function"]()