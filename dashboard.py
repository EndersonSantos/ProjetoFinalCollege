# Import
import pandas as pd
import streamlit as st
from gensim.models import Word2Vec
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

# Apply the theme
st.set_page_config(
    page_title="Decoding Movements Dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    menu_items={
        "About": "This app predicts the sector and territory of a financial instrument based on its description."
    },
)

# Use markdown for a custom-styled title and subtitle
st.markdown(
    "<h1 style='text-align: center; color: white;'>Decoding Movements</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h4 style='text-align: center; color: gray;'>Prever, através de descritivos, os setor e território de uma transação de um fundo de investimento.</h4>",
    unsafe_allow_html=True,
)
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
    user_input = st.text_area("", value=st.session_state["selected_example"], height=150)

    # Load pre-trained models and other necessary components based on the selected model
    if model_name:
        model_path = f'projetofinal/models/{model_name}.pkl'
        inverted_mapping_ter, inverted_mapping_sec, word2vec_model, clf_sec, clf_ter = read_pickle(model_path)

    # Predict button
    if st.button("Predict"):
        if user_input and model_name:
            tokens = user_input.lower().split()  # Simple tokenization, adjust as needed
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


    def try_read_csv(uploaded_file):
        encodings = ["utf-8", "iso-8859-1", "cp1252"]
        for enc in encodings:
            try:
                return pd.read_csv(uploaded_file, encoding=enc, delimiter=";"), enc
            except UnicodeDecodeError:
                continue
        return None, None


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

# Função para a página de análises
def analysis():
    st.title("Análises")
    st.write("Nesta página, é possível observar algumas análises sobre o Território e o Setor. No Território, apresentamos os cinco principais registos de cada país, enquanto no Setor disponibilizamos uma tabela com os registos mais frequentes. Tudo isto é realizado através do token.")

    # Carregar dados territoriais
    territorio_df = pd.read_csv("projetofinal/analysis/territorio_df.csv")

    # Carregar um shapefile do mundo
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    merged = world.merge(territorio_df, how='left', left_on='iso_a3', right_on='Território')

    # Criar um mapa folium centrado em uma localização inicial
    m = folium.Map(location=[0, 0], zoom_start=2)

# Função para lidar com o clique em um país
    

    
    # Adicionar o GeoDataFrame ao mapa folium
    folium.GeoJson(
        merged,
        name='geojson',
        tooltip=folium.GeoJsonTooltip(fields=['Descrição do território']),
        style_function=lambda feature: {'fillColor': '#0E7A0D' if feature['properties']['Território'] in territorio_df['Território'].values else '#ffffff', 'color': '#003300'},  # cor de preenchimento para países com tokens no top 5
        highlight_function=lambda feature: {'weight': 0},  # remove o destaque dos países
        popup=folium.Popup(lambda feature: click_handler(feature, territorio_df), max_width=400)  # aumentar a largura máxima da caixa de texto do pop-up
    ).add_to(m)
    
    # Função para lidar com o clique em um país
    def click_handler(feature, **kwargs):
        iso_a3 = feature['iso_a3']
        top5 = territorio_df[territorio_df['Território'] == iso_a3].nlargest(5, 'Contagem')
        popup_text = f"{feature['Descrição do território']} Top 5:\n"
        if not top5.empty:
            for _, row in top5.iterrows():
                popup_text += f"{row['Token']}:{row['Contagem'] }\n"
        return popup_text
    
     # Adicionar manipuladores de eventos de clique para cada país
    for _, feature in merged.iterrows():
        iso_a3 = feature['iso_a3']
        top5 = territorio_df[territorio_df['Território'] == iso_a3].nlargest(5, 'Contagem')
        if not top5.empty:
            geojson = folium.GeoJson(
                feature['geometry'],
                style_function=lambda feature: {'fillColor': '#0E7A0D', 'color': '#003300'},  # cor de preenchimento para países com tokens no top 5
                highlight_function=lambda feature: {'weight': 0},  # remove o destaque dos países
                name='geojson',
                popup=folium.Popup(click_handler(feature), max_width=400)  # aumentar a largura máxima da caixa de texto do pop-up
            )
        else:
            geojson = folium.GeoJson(
                feature['geometry'],
                style_function=lambda feature: {'fillColor': '#ffffff', 'color': '#000000'},  # remova a cor de preenchimento para países sem lista
                highlight_function=lambda feature: {'weight': 0},  # remove o destaque dos países
                name='geojson',
                popup=None  # sem caixa de texto para países sem lista
            )
        geojson.add_to(m)

    # Exibir o mapa interativo
    st.markdown("### Mapa - Análise por Território")
    folium_static(m)


    setor_df = pd.read_csv('projetofinal/analysis/setor_df.csv')
    setor_df1 = setor_df.drop(['Unnamed: 0','Descrição do Setor Institucional'], axis = 1)
    # Função para exibir todos os registros para um determinado setor
    def registros_por_setor(setor):
        return setor_df1[setor_df['Setor Institucional'] == setor]

    # Função para atualizar a lista de registros quando um novo setor é selecionado
    def atualizar_registros(setor_selecionado):
        st.write(f"Registros para {setor_selecionado} ({setor_df.loc[setor_df['Setor Institucional'] == setor_selecionado, 'Descrição do Setor Institucional'].iloc[0]}):")
        registros = registros_por_setor(setor_selecionado)
        st.write(registros)


    # Obter a lista única de setores institucionaiss
    st.markdown("### Análise por Setor")

    setores = setor_df1['Setor Institucional'].unique()

    # Criar o menu suspenso para selecionar o setor
    setor_selecionado = st.selectbox("Selecione o setor:", setores, index=0)

    # Definir a função de retorno de chamada para atualizar os registros quando um novo setor é selecionado
    atualizar_registros(setor_selecionado)

 



# Mapeia o nome das páginas às funções correspondentes

pages = {
    "Principal": {"function": home, "icon": "🏠"},
    "Análises": {"function": analysis, "icon": "📊"}
}

# Adicionando uma nova página "Análises" com ícone de análises no sidebar
selection = st.sidebar.radio("Go to", [f"{pages[key]['icon']} {key}" for key in pages.keys()], index=0)

# Chamando a função correspondente à página selecionada
page_name = selection.split()[1]  # Extraindo o nome da página do texto selecionado no sidebar
pages[page_name]["function"]()

