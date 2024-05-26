# %% [markdown]
# # Objective
# - O objetivo é treinar um modelo utilizando a descrição do produto para prever o setor e o território
# - Vamos utilizar Word2Vec para trasformar a descrição em vetores (embeddings)
# - Vamos utilizar um modelo XG boost para treinar

# %% [markdown]
# # Data
# - Dados fornecidos pelo banco de Portugal

# %% [markdown]
# # Imports

# %%
import pandas as pd
import numpy as np
import gradio as gr
import pickle
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from gensim.models import Word2Vec
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from feature_engine.encoding import RareLabelEncoder
from gensim.utils import simple_preprocess
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, precision_score, recall_score, roc_auc_score

# %% [markdown]
# # Reading Data

# %%
df = pd.read_excel('../../data/01.Dataset FI_06032024.xlsx', sheet_name=2)

# %% [markdown]
# # Small Preprocessing

# %% [markdown]
# Mapeando o código ao nome da descrição

# %%
MAP_TipoInformacao = {"A": "ativo", "P": "passivo"}
MAP_TipoInstrumento = {"F21": "Numerário", "F22": "Depósitos transferíveis", "F29": "Outros depósitos", "F3_P": "Títulos de dívida", "F4": "Empréstimos", "F511": "Ações cotadas", "F512": "Ações não cotadas", "F519": "Outras participações", "F521": "Unidades de Participação emitidas por FMM", "F522": "Unidades de Participação emitidas por FI, excluindo FMM", "F71": "Derivados financeiros"}
MAP_MaturidadeOriginal = {"01": "A vista", "10": "Ate 1 ano", "06": "De 1 a 2 anos", "07": "De 2 a 5 anos", "08": "A mais de 5 anos", "_Z": "Não aplicável"}

# %%
df.TipoInformacao = df.TipoInformacao.map(MAP_TipoInformacao)
df.TipoInstrumento = df.TipoInstrumento.map(MAP_TipoInstrumento)
df.MaturidadeOriginal = df.MaturidadeOriginal.map(MAP_MaturidadeOriginal)

# %%
df.drop(["CodEntidadeRef", "CodEntidadeCon"], axis=1, inplace=True)

# %%
df.head(3)

# %% [markdown]
# # Feature Engineering

# %%
df_clean = df.copy()

# %% [markdown]
# ### Encoding Rare Labels

# %% [markdown]
# ### Label Enconder

# %% [markdown]
# Criando funções para fazer o enconding dos targets

# %%
def encode_target(label, category_mapping):
  # Check if label is unseen (not in the dictionary)
  if label not in category_mapping:
    # Assign next available integer as seen in training data
    new_value = len(category_mapping)
    category_mapping[label] = new_value
  
  return category_mapping[label]

# %%
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

# %%
def return_map(df_clean):

    territory_map = {}
    sector_map = {}

    # Iterate through each row (assuming TerritorioCon and encoded_label_territorio are in the same order)
    for territorio, encoded_label in zip(df_clean["TerritorioCon"], df_clean["encoded_label_territorio"]):
      # Add the mapping to the dictionary if the TerritorioCon is not already present
      if territorio not in territory_map:
        territory_map[territorio] = encoded_label
    
    for sector, encoded_label_sector in zip(df_clean["SetorInstitucionalCon"], df_clean["encoded_label_setor"]):
      # Add the mapping to the dictionary if the TerritorioCon is not already present
      if sector not in sector_map:
        sector_map[sector] = encoded_label_sector

    return territory_map, sector_map

# %% [markdown]
# Aplicando funções para mapear os targets a códigos pra podermos treinar o modelo

# %%
# Get unique categories from 'TerritorioCon' column
unique_categories_ter = df_clean['TerritorioCon'].unique()
category_mapping_ter = dict(zip(unique_categories_ter, range(len(unique_categories_ter))))
inverted_mapping_ter = {value: key for key, value in category_mapping_ter.items()}

unique_categories_sec = df_clean["SetorInstitucionalCon"].unique()
category_mapping_sec = dict(zip(unique_categories_sec, range(len(unique_categories_sec))))
inverted_mapping_sec = {value: key for key, value in category_mapping_sec.items()}

df_clean["encoded_label_territorio"] = df_clean["TerritorioCon"].apply(encode_target, args=[category_mapping_ter])
df_clean["encoded_label_setor"] = df_clean['SetorInstitucionalCon'].apply(encode_target, args=[category_mapping_sec])

# %%
df_clean.head(3)

# %% [markdown]
# ### Processing Description Column

# %% [markdown]
# Vamos agora:
# 1. Utilizar a função simple_preprocess para aplicarmos tecnicas de Text Mining para limpar a descrição
# 2. Treinando o modelo Word2Vec para processar coluna com a descrição para vetores (Embeddings)

# %%
help(simple_preprocess)

# %%
df_clean['tokenized_Descricao_text'] = df_clean['DescricaoInstrumento'].apply(lambda x: simple_preprocess(x))
word2vec_model = Word2Vec(sentences=df_clean['tokenized_Descricao_text'], vector_size=100, window=5, min_count=1, workers=4)

# %%
df_clean

# %%
def compute_avg_embedding(tokens, unknown_embedding=[0]*word2vec_model.vector_size):
    embeddings = [word2vec_model.wv[token] for token in tokens if token in word2vec_model.wv]
    if embeddings:  # Embeddings found
        return np.array(embeddings).mean(axis=0)  # Return average embedding as a NumPy array
    else:  # No embeddings found
        return np.array(unknown_embedding)

# %%
df_clean['avg_embedding'] = df_clean['tokenized_Descricao_text'].apply(compute_avg_embedding)
X = df_clean['avg_embedding'].apply(pd.Series).to_numpy()
y1 = df_clean['encoded_label_territorio']
y2 = df_clean['encoded_label_setor']
embed_data = pd.DataFrame(X)

# %%
df_clean.head(3)

# %% [markdown]
# ### Train Test Split

# %% [markdown]
# Separando os dados em treino e test. Vamos ter dois targets. Um para o setor e outro para o território

# %%
# Stratified split with 'TerritorioCon' as the stratification factor
#df_train, df_test = train_test_split(embed_data, test_size=0.2, random_state=42)
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X, y1, test_size=0.1, random_state=41)
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X, y2, test_size=0.1, random_state=41)

# %% [markdown]
# # Train 

# %% [markdown]
# ## XG Boost

# %% [markdown]
# ## Train the Model Territorio

# %%
clf_ter = XGBClassifier(random_state=42, max_depth=4)
clf_ter.fit(X_train_1, y_train_1, early_stopping_rounds=10, 
        eval_set=[(X_test_1, y_test_1)])

# %% [markdown]
# ## Evaluate 

# %%
y_pred_test = clf_ter.predict(X_test_1)
y_pred_train = clf_ter.predict(X_train_1)

# %%
pred_test_str = map_numbers_to_categories(y_pred_test, inverted_mapping_ter)
pred_train_str = map_numbers_to_categories(y_pred_train, inverted_mapping_ter)

# %%
model_names = []

territorio_accuracy_treino = []
territorio_precision_treino = []
territorio_recall_treino = []

territorio_accuracy_test = []
territorio_precision_test = []
territorio_recall_test = []

# %%
model_names.append("XG-Boost")

accuracy_test = accuracy_score(y_test_1.to_list(), y_pred_test)
accuracy_train = accuracy_score(y_train_1.to_list(), y_pred_train)
territorio_accuracy_treino.append(accuracy_train)
territorio_accuracy_test.append(accuracy_test)


precision_test = precision_score(y_test_1.to_list(), y_pred_test, average='weighted')
precision_train = precision_score(y_train_1.to_list(), y_pred_train, average='weighted')
territorio_precision_treino.append(precision_train)
territorio_precision_test.append(precision_test)


recall_test = recall_score(y_test_1.to_list(), y_pred_test, average='weighted')
recall_train = recall_score(y_train_1.to_list(), y_pred_train, average='weighted')
territorio_recall_treino.append(recall_train)
territorio_recall_test.append(recall_test)

# %%
# Evaluate accuracy
print("Accuracy Test:", accuracy_test)
print("Accuracy Train:", accuracy_train)

# Precision
print("Precision Test:", precision_test)
print("Precision Train:", precision_train)

# Recall
print("Recall Test:", recall_test)
print("Recall Train:", recall_train)

# %%
print("Classification Report:")
print(classification_report(y_test_1.to_list(), y_pred_test))

# %% [markdown]
# ## Train Model Sector

# %%
clf_sec = XGBClassifier(random_state=42)
clf_sec.fit(X_train_2, y_train_2, early_stopping_rounds=10, 
        eval_set=[(X_test_2, y_test_2)])

# %%
y_pred_test = clf_sec.predict(X_test_2)
y_pred_train = clf_sec.predict(X_train_2)

# %%
pred_test_str = map_numbers_to_categories(y_pred_test, inverted_mapping_sec)
pred_train_str = map_numbers_to_categories(y_pred_train, inverted_mapping_sec)

# %%
setor_accuracy_treino = []
setor_precision_treino = []
setor_recall_treino = []

setor_accuracy_test = []
setor_precision_test = []
setor_recall_test = []

# %%
accuracy_test = accuracy_score(y_test_2.to_list(), y_pred_test)
accuracy_train = accuracy_score(y_train_2.to_list(), y_pred_train)
setor_accuracy_treino.append(accuracy_train)
setor_accuracy_test.append(accuracy_test)


precision_test = precision_score(y_test_2.to_list(), y_pred_test, average='weighted')
precision_train = precision_score(y_train_2.to_list(), y_pred_train, average='weighted')
setor_precision_treino.append(precision_train)
setor_precision_test.append(precision_test)


recall_test = recall_score(y_test_2.to_list(), y_pred_test, average='weighted')
recall_train = recall_score(y_train_2.to_list(), y_pred_train, average='weighted')
setor_recall_treino.append(recall_train)
setor_recall_test.append(recall_test)

# %%
# Evaluate accuracy
print("Accuracy Test:", accuracy_test)
print("Accuracy Train:", accuracy_train)

# Precision
print("Precision Test:", precision_test)
print("Precision Train:", precision_train)

# Recall
print("Recall Test:", recall_test)
print("Recall Train:", recall_train)

# %%
print("Classification Report:")
print(classification_report(y_test_2.to_list(), y_pred_test))

# %% [markdown]
# ## Train more models

# %%
modelos = {
    'Decision Tree': DecisionTreeClassifier(max_depth=3),
    'KNN': KNeighborsClassifier(n_neighbors=5),  # Ajuste o valor de k
    'Regressão Logística': LogisticRegression(multi_class='multinomial'),
    'SVM':  SVC()
}

# %%
# Treinar e avaliar cada modelo
for nome_modelo, modelo in modelos.items():
    
    # Treinar o modelo
    modelo.fit(X_train_1, y_train_1)
    
    y_pred_test = modelo.predict(X_test_1)
    y_pred_train = modelo.predict(X_train_1)
    
    model_names.append(nome_modelo)
    
    accuracy_test = accuracy_score(y_test_1.to_list(), y_pred_test)
    accuracy_train = accuracy_score(y_train_1.to_list(), y_pred_train)
    territorio_accuracy_treino.append(accuracy_train)
    territorio_accuracy_test.append(accuracy_test)


    precision_test = precision_score(y_test_1.to_list(), y_pred_test, average='weighted')
    precision_train = precision_score(y_train_1.to_list(), y_pred_train, average='weighted')
    territorio_precision_treino.append(precision_train)
    territorio_precision_test.append(precision_test)


    recall_test = recall_score(y_test_1.to_list(), y_pred_test, average='weighted')
    recall_train = recall_score(y_train_1.to_list(), y_pred_train, average='weighted')
    territorio_recall_treino.append(recall_train)
    territorio_recall_test.append(recall_test)

# %%
# Treinar e avaliar cada modelo
for nome_modelo, modelo in modelos.items():
    
    # Treinar o modelo
    modelo.fit(X_train_2, y_train_2)
    
    y_pred_test = modelo.predict(X_test_2)
    y_pred_train = modelo.predict(X_train_2)
    
    
    accuracy_test = accuracy_score(y_test_2.to_list(), y_pred_test)
    accuracy_train = accuracy_score(y_train_2.to_list(), y_pred_train)
    setor_accuracy_treino.append(accuracy_train)
    setor_accuracy_test.append(accuracy_test)


    precision_test = precision_score(y_test_2.to_list(), y_pred_test, average='weighted')
    precision_train = precision_score(y_train_2.to_list(), y_pred_train, average='weighted')
    setor_precision_treino.append(precision_train)
    setor_precision_test.append(precision_test)


    recall_test = recall_score(y_test_2.to_list(), y_pred_test, average='weighted')
    recall_train = recall_score(y_train_2.to_list(), y_pred_train, average='weighted')
    setor_recall_treino.append(recall_train)
    setor_recall_test.append(recall_test)

# %% [markdown]
# ## Plot da Performance dos Modelos

# %%
territorio_performance_df = pd.DataFrame({"Accuracy Treino": territorio_accuracy_treino,"Accuracy Test": territorio_accuracy_test, "Precision Treino": territorio_precision_treino, "Precision Test": territorio_precision_test, "Recall Treino": territorio_recall_treino, "Recall Test": territorio_recall_test}, index=model_names)
territorio_performance_df= territorio_performance_df.round(2)
territorio_performance_df

# %%
# Reset the DataFrame index
df_reseted = territorio_performance_df.reset_index()

# Define colors for each model
colors = ['blue', 'orange']  # Adjust colors as needed

# Create subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 18))  # Adjust figure size if necessary

# Define the metrics and corresponding labels
metrics = ['Accuracy', 'Precision', 'Recall']
y_labels = ['Model Accuracy', 'Model Precision', 'Model Recall']

for i, metric in enumerate(metrics):
    # Plot the metric for both training and test sets
    bars = df_reseted.plot(kind='bar', x='index', y=[f'{metric} Treino', f'{metric} Test'], stacked=False, color=colors, ax=axs[i])
    
    # Customize the chart
    axs[i].set_title(f'{metric}: Comparação dos Modelos - Território')
    axs[i].set_xlabel('Model')
    axs[i].set_ylabel(y_labels[i])
    axs[i].legend(title='Data', loc=(1.1, 0.5))
    axs[i].set_xticklabels(df_reseted['index'], rotation=45, ha='right')  # Rotate x-axis labels for better readability

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()

# %%
setor_performance_df = pd.DataFrame({"Accuracy Treino": setor_accuracy_treino,"Accuracy Test": setor_accuracy_test, "Precision Treino": setor_precision_treino, "Precision Test": setor_precision_test, "Recall Treino": setor_recall_treino, "Recall Test": setor_recall_test}, index=model_names)
setor_performance_df.round(2)

# %%
# Reset the DataFrame index
df_reseted = setor_performance_df.reset_index()

# Define colors for each model
colors = ['blue', 'orange']  # Adjust colors as needed

# Create subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 18))  # Adjust figure size if necessary

# Define the metrics and corresponding labels
metrics = ['Accuracy', 'Precision', 'Recall']
y_labels = ['Model Accuracy', 'Model Precision', 'Model Recall']

for i, metric in enumerate(metrics):
    # Plot the metric for both training and test sets
    bars = df_reseted.plot(kind='bar', x='index', y=[f'{metric} Treino', f'{metric} Test'], stacked=False, color=colors, ax=axs[i])
    
    # Customize the chart
    axs[i].set_title(f'{metric}: Comparação dos Modelos - Setor')
    axs[i].set_xlabel('Model')
    axs[i].set_ylabel(y_labels[i])
    axs[i].legend(title='Data', loc=(1.1, 0.5))
    axs[i].set_xticklabels(df_reseted['index'], rotation=45, ha='right')  # Rotate x-axis labels for better readability

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()

# %% [markdown]
# ## Treinar o melhor modelo com mais Atributos

# %%
new_features = ["TipoInformacao", "TipoInstrumento", "MaturidadeOriginal"]
df_to_get_dummies = df_clean[new_features]
dummies = pd.get_dummies(df_to_get_dummies)

new_df_to_train = pd.concat([embed_data, dummies], axis=1)

# %%
# Stratified split with 'TerritorioCon' as the stratification factor
#df_train, df_test = train_test_split(embed_data, test_size=0.2, random_state=42)
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(new_df_to_train, y1, test_size=0.1, random_state=41)
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(new_df_to_train, y2, test_size=0.1, random_state=41)

# %% [markdown]
# ## Train 

# %% [markdown]
# ## Train the Model Territorio

# %%
clf_ter = XGBClassifier(random_state=42, max_depth=4)
clf_ter.fit(X_train_1, y_train_1, early_stopping_rounds=10, 
        eval_set=[(X_test_1, y_test_1)])

# %% [markdown]
# ## Evaluate 

# %%
y_pred_test = clf_ter.predict(X_test_1)
y_pred_train = clf_ter.predict(X_train_1)

pred_test_str = map_numbers_to_categories(y_pred_test, inverted_mapping_ter)
pred_train_str = map_numbers_to_categories(y_pred_train, inverted_mapping_ter)

# %%
accuracy_test = accuracy_score(y_test_1.to_list(), y_pred_test)
accuracy_train = accuracy_score(y_train_1.to_list(), y_pred_train)

precision_train = precision_score(y_train_1.to_list(), y_pred_train, average='weighted')
precision_test = precision_score(y_test_1.to_list(), y_pred_test, average='weighted')

recall_test = recall_score(y_test_1.to_list(), y_pred_test, average='weighted')
recall_train = recall_score(y_train_1.to_list(), y_pred_train, average='weighted')

# %%
accuracy_train

# %%
accuracy_test

# %%
models_names = ["XG_Boost", "XG Boost - More Features"]
accuracy_treino_ter = [territorio_accuracy_treino[0], accuracy_train]
accuracy_test_ter = [territorio_accuracy_test[0], accuracy_test]

precision_treino_ter = [territorio_precision_treino[0], precision_train]
precision_test_ter = [territorio_precision_test[0], precision_test]

recall_treino_ter = [territorio_recall_treino[0], recall_train]
recall_test_ter = [territorio_recall_test[0], recall_test]

# %%
data_performance = pd.DataFrame({"Accuracy Treino": accuracy_treino_ter,"Accuracy Test": accuracy_test_ter, "Precision Treino": precision_treino_ter, "Precision Test": precision_test_ter, "Recall Treino": recall_treino_ter, "Recall Test": recall_test_ter}, index=models_names).T
data_performance["uplift"] = (data_performance["XG Boost - More Features"] - data_performance["XG_Boost"]) / data_performance["XG_Boost"]
data_performance['uplift'] = data_performance['uplift'].apply(lambda x: '{:.2%}'.format(x))
data_performance.round(2)
data_performance

# %%
# Preparing the data
metrics = ['Accuracy', 'Precision', 'Recall']
types = ['Treino', 'Test']
x = np.arange(len(metrics))  # the label locations

fig, ax = plt.subplots(figsize=(10, 6))

# Plotting the data for both models
for i, type in enumerate(types):
    xg_boost_scores = data_performance.loc[[f'{metric} {type}' for metric in metrics], 'XG_Boost']
    xg_boost_more_features_scores = data_performance.loc[[f'{metric} {type}' for metric in metrics], 'XG Boost - More Features']
    
    # Line plots for each model
    line1, = ax.plot(x + i*0.2, xg_boost_scores, marker='o', linestyle='-', label=f'XG_Boost {type}')
    line2, = ax.plot(x + i*0.2, xg_boost_more_features_scores, marker='s', linestyle='-', label=f'XG Boost - More Features {type}')
    
    # Annotating the uplift for each metric
    for j, metric in enumerate(metrics):
        uplift = data_performance.loc[f'{metric} {type}', 'uplift']
        higher_score = max(xg_boost_scores.iloc[j], xg_boost_more_features_scores.iloc[j])
        ax.text(x[j] + i*0.2, higher_score + 0.01, f'{uplift}', ha='center', fontsize=9, bbox=dict(facecolor='white', edgecolor='none', pad=2))

# Adding some text for labels, title, and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Model Performance and Uplift by Metric and Type')
ax.set_xticks(x + 0.1)  # Adjusting x-ticks to be in the middle of the groups
ax.set_xticklabels(metrics)
ax.legend()
ax.set_ylim(0.8, 1)

fig.tight_layout()
plt.show()

# %%
print("Classification Report:")
print(classification_report(y_test_1.to_list(), y_pred_test))

# %% [markdown]
# ## Train Model Sector

# %%
clf_sec = XGBClassifier(random_state=42)
clf_sec.fit(X_train_2, y_train_2, early_stopping_rounds=10, 
        eval_set=[(X_test_2, y_test_2)])

# %%
y_pred_test = clf_sec.predict(X_test_2)
y_pred_train = clf_sec.predict(X_train_2)

pred_test_str = map_numbers_to_categories(y_pred_test, inverted_mapping_sec)
pred_train_str = map_numbers_to_categories(y_pred_train, inverted_mapping_sec)

# %%
accuracy_test = accuracy_score(y_test_2.to_list(), y_pred_test)
accuracy_train = accuracy_score(y_train_2.to_list(), y_pred_train)

precision_test = precision_score(y_test_2.to_list(), y_pred_test, average='weighted')
precision_train = precision_score(y_train_2.to_list(), y_pred_train, average='weighted')

recall_test = recall_score(y_test_2.to_list(), y_pred_test, average='weighted')
recall_train = recall_score(y_train_2.to_list(), y_pred_train, average='weighted')

# %%
models_names = ["XG_Boost", "XG Boost - More Features"]
accuracy_treino_sec = [setor_accuracy_treino[0], accuracy_train]
accuracy_test_sec = [setor_accuracy_test[0], accuracy_test]

precision_treino_sec = [setor_precision_treino[0], precision_train]
precision_test_sec = [setor_precision_test[0], precision_test]

recall_treino_sec = [setor_recall_treino[0], recall_train]
recall_test_sec = [setor_recall_test[0], recall_test]

# %%
data_performance = pd.DataFrame({"Accuracy Treino": accuracy_treino_sec,"Accuracy Test": accuracy_test_sec, "Precision Treino": precision_treino_sec, "Precision Test": precision_test_sec, "Recall Treino": recall_treino_sec, "Recall Test": recall_test_sec}, index=models_names).T
data_performance["uplift"] = (data_performance["XG Boost - More Features"] - data_performance["XG_Boost"]) / data_performance["XG_Boost"]
data_performance['uplift'] = data_performance['uplift'].apply(lambda x: '{:.2%}'.format(x))
data_performance = data_performance.round(2)
data_performance

# %%
# Preparing the data
metrics = ['Accuracy', 'Precision', 'Recall']
types = ['Treino', 'Test']
x = np.arange(len(metrics))  # the label locations

fig, ax = plt.subplots(figsize=(10, 6))

# Plotting the data for both models
for i, type in enumerate(types):
    xg_boost_scores = data_performance.loc[[f'{metric} {type}' for metric in metrics], 'XG_Boost']
    xg_boost_more_features_scores = data_performance.loc[[f'{metric} {type}' for metric in metrics], 'XG Boost - More Features']
    
    # Line plots for each model
    line1, = ax.plot(x + i*0.2, xg_boost_scores, marker='o', linestyle='-', label=f'XG_Boost {type}')
    line2, = ax.plot(x + i*0.2, xg_boost_more_features_scores, marker='s', linestyle='-', label=f'XG Boost - More Features {type}')
    
    # Annotating the uplift for each metric
    for j, metric in enumerate(metrics):
        uplift = data_performance.loc[f'{metric} {type}', 'uplift']
        higher_score = max(xg_boost_scores.iloc[j], xg_boost_more_features_scores.iloc[j])
        ax.text(x[j] + i*0.2, higher_score + 0.01, f'{uplift}', ha='center', fontsize=9, bbox=dict(facecolor='white', edgecolor='none', pad=2))

# Adding some text for labels, title, and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Model Performance and Uplift by Metric and Type')
ax.set_xticks(x + 0.1)  # Adjusting x-ticks to be in the middle of the groups
ax.set_xticklabels(metrics)
ax.legend()
ax.set_ylim(0.8, 1)

fig.tight_layout()
plt.show()

# %%
print("Classification Report:")
print(classification_report(y_test_2.to_list(), y_pred_test))

# %% [markdown]
# # Analysing results that failed

# %% [markdown]
# - HRV
# - 'IRN'
# - '5D'
# - 

# %%
df_clean[df_clean.TerritorioCon == 'GBR']

# %%
#return_embeedings('AFDB 5.75 Perp')
return_embeedings('Carlyle Europ PatIII', word2vec_model, simple_preprocess, clf_ter, clf_sec, inverted_mapping_ter, inverted_mapping_sec)

# %%


# %%


# %% [markdown]
# # Training with the whole data

# %% [markdown]
# Vamos treinar agora o modelo com os dados todos. São exatamente os mesmos passos aplicados aos dados todos

# %%
df = pd.read_excel('../../data/01.Dataset FI_06032024.xlsx', sheet_name=2)

# %%
df_clean = df.copy("../data_missing/sampled_indices.csv")

# %%
data_eval = df_clean.sample(1000, random_state=42)
not_sampled_mask = ~df_clean.index.isin(data_eval)  # Create a boolean mask for non-sampled rows
df_clean = df_clean[not_sampled_mask]


data_pred = df_clean.sample(1000, random_state=42)
not_sampled_mask = ~df_clean.index.isin(data_pred)
df_clean = df_clean[not_sampled_mask]

# %%
df_clean.to_csv("../data_train/data_train.csv")
data_eval.to_csv("../data_train/data_eval.csv")
data_pred.to_csv("../data_missing/data_pred.csv")

# %%
# Get unique categories from 'TerritorioCon' column
unique_categories_ter = df_clean['TerritorioCon'].unique()
category_mapping_ter = dict(zip(unique_categories_ter, range(len(unique_categories_ter))))
inverted_mapping_ter = {value: key for key, value in category_mapping_ter.items()}

unique_categories_sec = df_clean["SetorInstitucionalCon"].unique()
category_mapping_sec = dict(zip(unique_categories_sec, range(len(unique_categories_sec))))
inverted_mapping_sec = {value: key for key, value in category_mapping_sec.items()}

df_clean["encoded_label_territorio"] = df_clean["TerritorioCon"].apply(encode_target, args=[category_mapping_ter])
df_clean["encoded_label_setor"] = df_clean['SetorInstitucionalCon'].apply(encode_target, args=[category_mapping_sec])

# %%
df_clean['tokenized_Descricao_text'] = df_clean['DescricaoInstrumento'].apply(lambda x: simple_preprocess(x))
word2vec_model = Word2Vec(sentences=df_clean['tokenized_Descricao_text'], vector_size=100, window=5, min_count=1, workers=4)

# %%
df_clean['tokenized_Descricao_text'] = df_clean['DescricaoInstrumento'].apply(lambda x: simple_preprocess(x))
word2vec_model = Word2Vec(sentences=df_clean['tokenized_Descricao_text'], vector_size=100, window=5, min_count=1, workers=4)

df_clean['avg_embedding'] = df_clean['tokenized_Descricao_text'].apply(compute_avg_embedding)
X = df_clean['avg_embedding'].apply(pd.Series).to_numpy()
y1 = df_clean['encoded_label_territorio']
y2 = df_clean['encoded_label_setor']
embed_data = pd.DataFrame(X)

# %%
df_clean.head()

# %%
new_features = ["TipoInformacao", "TipoInstrumento", "MaturidadeOriginal"]
df_to_get_dummies = df_clean[new_features]
dummies = pd.get_dummies(df_to_get_dummies)

new_df_to_train = pd.concat([embed_data, dummies], axis=1)

# %%
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X, y1, test_size=0.2, random_state=42)
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X, y2, test_size=0.2, random_state=42)

# %%
clf_ter = XGBClassifier(random_state=42, max_depth=5)
clf_ter.fit(X, y1)

# %%
clf_sec = XGBClassifier(random_state=42, max_depth=5)
clf_sec.fit(X, y2)

# %%
y_pred_test = clf_ter.predict(X)
pred_test_str = map_numbers_to_categories(y_pred_test, inverted_mapping_ter)

# Evaluate accuracy
accuracy_test = accuracy_score(y1.to_list(), y_pred_test)
print("Accuracy Ter:", accuracy_test)

# %%
print("Classification Report:")
print(classification_report(y1.to_list(), y_pred_test))

# %%
y_pred_test = clf_sec.predict(X)
pred_test_str = map_numbers_to_categories(y_pred_test, inverted_mapping_sec)

# Evaluate accuracy
accuracy_test = accuracy_score(y2.to_list(), y_pred_test)
print("Accuracy Sec:", accuracy_test)

# %%
print("Classification Report:")
print(classification_report(y2.to_list(), y_pred_test))

# %%
y_pred_test = clf_ter.predict_proba(X)

# %%
y_pred_test = clf_sec.predict_proba(X)

# %%
y_pred_test.max(axis=1).min()

# %%
clf_ter.predict_proba(X[:1]).max()

# %%
!ls ../final_data

# %%
pred_with_prob = pd.read_csv("../final_data/df_final.csv")

pred_with_prob[pred_with_prob["ter_probabilidade"] < 0.5]

# %% [markdown]
# ## Mais Features no dataset final e avaliar performance

# %%
df_clean = df.copy("../data_missing/sampled_indices.csv")

# %%
# Get unique categories from 'TerritorioCon' column
unique_categories_ter = df_clean['TerritorioCon'].unique()
category_mapping_ter = dict(zip(unique_categories_ter, range(len(unique_categories_ter))))
inverted_mapping_ter = {value: key for key, value in category_mapping_ter.items()}

unique_categories_sec = df_clean["SetorInstitucionalCon"].unique()
category_mapping_sec = dict(zip(unique_categories_sec, range(len(unique_categories_sec))))
inverted_mapping_sec = {value: key for key, value in category_mapping_sec.items()}

df_clean["encoded_label_territorio"] = df_clean["TerritorioCon"].apply(encode_target, args=[category_mapping_ter])
df_clean["encoded_label_setor"] = df_clean['SetorInstitucionalCon'].apply(encode_target, args=[category_mapping_sec])

# %%
df_clean['tokenized_Descricao_text'] = df_clean['DescricaoInstrumento'].apply(lambda x: simple_preprocess(x))
word2vec_model = Word2Vec(sentences=df_clean['tokenized_Descricao_text'], vector_size=100, window=5, min_count=1, workers=4)

# %%
df_clean['avg_embedding'] = df_clean['tokenized_Descricao_text'].apply(compute_avg_embedding)
X = df_clean['avg_embedding'].apply(pd.Series).to_numpy()
y1 = df_clean['encoded_label_territorio']
y2 = df_clean['encoded_label_setor']
embed_data = pd.DataFrame(X)

new_features = ["TipoInformacao", "TipoInstrumento", "MaturidadeOriginal"]
df_to_get_dummies = df_clean[new_features]
dummies = pd.get_dummies(df_to_get_dummies)

new_df_to_train = pd.concat([embed_data, dummies], axis=1)

# %%
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(new_df_to_train, y1, test_size=0.2, random_state=42)
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(new_df_to_train, y2, test_size=0.2, random_state=42)

# %%
clf_ter = XGBClassifier(random_state=42, max_depth=5)
clf_ter.fit(new_df_to_train, y1)

# %%
clf_sec = XGBClassifier(random_state=42, max_depth=5)
clf_sec.fit(new_df_to_train, y2)

# %%
y_pred_test = clf_ter.predict(new_df_to_train)
pred_test_str = map_numbers_to_categories(y_pred_test, inverted_mapping_ter)

# Evaluate accuracy
accuracy_test = accuracy_score(y1.to_list(), y_pred_test)
print("Accuracy Ter:", accuracy_test)

# %%
y_pred_test = clf_sec.predict(new_df_to_train)
pred_test_str = map_numbers_to_categories(y_pred_test, inverted_mapping_sec)

# Evaluate accuracy
accuracy_test = accuracy_score(y2.to_list(), y_pred_test)
print("Accuracy Sec:", accuracy_test)

# %% [markdown]
# ## Fazer
# - Adicionar mais features no dataset final e avaliar 
# - Adicionar predict probability
# - Rever diretório para onde data deve está 
# - Rever o Read ME
# - Analisar casos que falharam
# - Organizar Notebook 
# - Fazer fix da ci/cd
# - Fazer o push da imagem do ci/cd
# - colacar dados na s3 e criar logica para acessa dados da s3
# - Fazer o push do modelo para um bucket e acessar modelo de lá para utilizar
# - Criar a app no ECS 
# - Escrever relarório -> 
#     - Treinamos com mais features e melhorou
#     - Mas não melhorou
#     - Dizer que separamos em treino e teste mas no final treinamos com os dados todos
#     - Sugestão de como se pode melhorar
#     - Explicar que treinamos dois modelos
# - Questions
#     - How predict proba works
#     - How the word2vec works 
#     - How simple process works
#     - How decision trees works
#     - How XG Boost Works
#     

# %% [markdown]
# Função que permite usar o modelo e retornar as previsões para novas entradas

# %%
def return_embeedings(string, word2vec_model, simple_preprocess, clf_t, clf_s, inverted_mapping_t, inverted_mapping_s):
    case = pd.DataFrame({'DescricaoInstrumento': [string]})
    case['tokenized_Descricao_text'] = case['DescricaoInstrumento'].apply(lambda x: simple_preprocess(x))
    case['avg_embedding'] = case['tokenized_Descricao_text'].apply(compute_avg_embedding)
    
    X = case['avg_embedding'].apply(pd.Series).to_numpy()
    
    prediction_t = clf_t.predict(X)
    str_pred_t = map_numbers_to_categories(prediction_t, inverted_mapping_t)
    
    prediction_s = clf_s.predict(X)
    str_pred_s = map_numbers_to_categories(prediction_s, inverted_mapping_s)
    
    prediction_s_prob = clf_sec.predict_proba(X).max()
    prediction_t_prob = clf_ter.predict_proba(X).max()
    
    return str_pred_t, str_pred_s #, prediction_t_prob, prediction_s_prob

# %%
str_to_try = 'HEATHROW FUNDING LTD 1.50% 12/10/2027'
str_pred_t, str_pred_s, prediction_s_prob, prediction_t_prob = return_embeedings(str_to_try, word2vec_model, simple_preprocess, clf_ter, clf_sec, inverted_mapping_ter, inverted_mapping_sec)

# %%
return_embeedings(str_to_try, word2vec_model, simple_preprocess, clf_ter, clf_sec, inverted_mapping_ter, inverted_mapping_sec)

# %%
all_str_ter = []
all_str_sec = []

for i in range(len(sampled_indices)):

    string = sampled_indices.DescricaoInstrumento.iloc[i]
    str_pred_t, str_pred_s = return_embeedings(string, word2vec_model, simple_preprocess, clf_ter, clf_sec, inverted_mapping_ter, inverted_mapping_sec)
    
    all_str_ter.append(str_pred_t[0])
    all_str_sec.append(str_pred_s[0])
    
sampled_indices["sec_pred"] = all_str_sec  
sampled_indices["ter_pred"] = all_str_ter

# %% [markdown]
# ## Gradio App

# %% [markdown]
# Criando um aplicativo gradio para demostrar o trabalho

# %%
def return_embeedings_(string):
    case = pd.DataFrame({'DescricaoInstrumento': [string]})
    case['tokenized_Descricao_text'] = case['DescricaoInstrumento'].apply(lambda x: simple_preprocess(x))
    case['avg_embedding'] = case['tokenized_Descricao_text'].apply(compute_avg_embedding)

    X = case['avg_embedding'].apply(pd.Series).to_numpy()

    try:
        prediction_t = clf_ter.predict(X)
        str_pred_t = map_numbers_to_categories(prediction_t, inverted_mapping_ter)

        prediction_s = clf_sec.predict(X)
        str_pred_s = map_numbers_to_categories(prediction_s, inverted_mapping_sec)
        prediction_s_prob = clf_sec.predict_proba(X).max()
        prediction_t_prob = clf_ter.predict_proba(X).max()

        return str(str_pred_t[0]), str(str_pred_s[0]) #, prediction_t_prob, prediction_s_prob
           
    except Exception as e:
        return "An error occurred: " + str(e)

# %%
examples = [
    ["HEATHROW FUNDING LTD 1.50% 12/10/2027"],
    ["FORTUM 1.625% A:27/02/2026"],
    ["BAC FLOAT 25/4/24"],
    ["DP DP 4M 0% 16/12/21CGD 0.00% 2020-12-16"],
]

# Create the Gradio app
iface = gr.Interface(
    # Argument 1: function (Required)
    fn=return_embeedings_,  # The function you want to expose as an interface

    # Argument 2: input components (Required)
    inputs=[gr.Textbox(lines=5, placeholder="Enter a text description")],  # Defines user input

    # Argument 3: examples (Optional)
    examples=examples,  # List of example text descriptions (or tuples with predictions)

    # Argument 4: output components (Required)
    outputs=[
        gr.Textbox(label="Prediction Territory"),
        gr.Textbox(label="Prediction Sector"),
        #gr.Textbox(label="Prediction Probability Territory"),
        #gr.Textbox(label="Prediction Probability Sector"),
    ],  # Defines how to display the output
)

# Launch the app
iface.launch()

# %% [markdown]
# ## Saving and reading models

# %%
with open('model_data.pkl', 'wb') as f:
    pickle.dump(inverted_mapping_ter, f)
    pickle.dump(inverted_mapping_sec, f)
    pickle.dump(word2vec_model, f)
    pickle.dump(clf_sec, f)
    pickle.dump(clf_ter, f)

# %%
with open('model_data.pkl', 'rb') as f:
    # Load the data objects in the same order they were saved
    inverted_mapping_ter = pickle.load(f)
    inverted_mapping_sec = pickle.load(f)
    word2vec_model = pickle.load(f)
    clf_sec = pickle.load(f)
    clf_ter = pickle.load(f)


