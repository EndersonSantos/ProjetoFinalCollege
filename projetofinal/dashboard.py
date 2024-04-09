#Import
import pandas as pd
import streamlit as st
from gensim.models import Word2Vec
import pickle
from preprocessing import compute_avg_embedding, read_pickle
from train_tools import return_embeedings

# Apply the theme
st.set_page_config(
    page_title="Decoding Movements Dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    menu_items={
        'About': "This app predicts the sector and territory of a financial instrument based on its description."
    }
)

# Use markdown for a custom-styled title and subtitle
st.markdown("<h1 style='text-align: center; color: white;'>Decoding Movements</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Prever, através de descritivos, os setor e território de uma transação de um fundo de investimento.</h4>", unsafe_allow_html=True)

# Initialize session state
if 'selected_example' not in st.session_state:
    st.session_state['selected_example'] = ""

# Text input for description
st.subheader("Description")
user_input = st.text_area("", value=st.session_state['selected_example'], height=150)

# Load pre-trained models and other necessary components
inverted_mapping_ter, inverted_mapping_sec, word2vec_model, clf_sec, clf_ter = read_pickle()

# Predict button
if st.button('Predict'):
    if user_input:
        tokens = user_input.lower().split()  # Simple tokenization, adjust as needed
        avg_embedding = compute_avg_embedding(tokens, word2vec_model)
        pred_territory, pred_sector = return_embeedings(
            user_input, word2vec_model, clf_ter, clf_sec, inverted_mapping_ter, inverted_mapping_sec
        )
        st.success(f'Predicted Territory: {pred_territory}')
        st.success(f'Predicted Sector: {pred_sector}')
    else:
        st.error('Please enter a description.')

# Quick example buttons
st.write("Quick Examples:")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("KFW 2.875% 29/05/26"):
        st.session_state['selected_example'] = "KFW 2.875% 29/05/26"
with col2:
    if st.button("HEIBOS 0.25 13/10/24"):
        st.session_state['selected_example'] = "HEIBOS 0.25 13/10/24"
with col3:
    if st.button("FIS 0.125 03/12/22"):
        st.session_state['selected_example'] = "FIS 0.125 03/12/22"

# File upload for prediction
st.header("Predict from File")
uploaded_file = st.file_uploader("Upload your file (CSV or Excel)", type=['csv', 'xlsx'], key="file-uploader")


def try_read_csv(uploaded_file):
    encodings = ['utf-8', 'iso-8859-1', 'cp1252']
    for enc in encodings:
        try:
            return pd.read_csv(uploaded_file, encoding=enc, delimiter=';'), enc
        except UnicodeDecodeError:
            continue
    return None, None 
        

# Function to process and predict the uploaded file
def process_and_predict_file(uploaded_file):
    # Read the uploaded file
    if uploaded_file is not None:
        if uploaded_file.type == "text/csv":
            df, encoding_used = try_read_csv(uploaded_file)
            if df is not None:
                st.success(f"File read successfully with encoding: {encoding_used}")
            else:
                st.error("Failed to read the file with common encodings. Please check the file encoding.")
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')

        # Initialize columns for predictions
        df['Predicted Territory'] = ""
        df['Predicted Sector'] = ""
        
        # Preprocess and predict for each row
        for index, row in df.iterrows():
            description = row['DescricaoInstrumento']
            # Assuming you have a tokenization and embedding process
            # similar to what's done for a single string prediction
            tokens = description.lower().split()  # Example tokenization, adjust as needed
            avg_embedding = compute_avg_embedding(tokens, word2vec_model)
            
            # Making predictions
            pred_territory, pred_sector = return_embeedings(
                description, word2vec_model, clf_ter, clf_sec, inverted_mapping_ter, inverted_mapping_sec
            )
            
            # Assigning predictions to the dataframe
            df.at[index, 'Predicted Territory'] = pred_territory
            df.at[index, 'Predicted Sector'] = pred_sector

        return df


# If a file was uploaded, process it and allow the user to download the results
if uploaded_file is not None:
    processed_df = process_and_predict_file(uploaded_file)
    st.write(processed_df)
    
    # Convert DataFrame to CSV for downloading
    csv = processed_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download predictions as CSV",
        data=csv,
        file_name='predictions.csv',
        mime='text/csv',
    )