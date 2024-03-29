# ProjetoFinalCollege
1. Instale todas as dependencias para o projeto: 
```
pip install -r requirements.txt
```
ps: Certifique que está no diretório certo

2. Adicione os dados necessários
- Para treinar o modelo coloque dentro do folder **projetofinal/data_train** adicione os dados que utilizará para treinar o modelo
- Adicione o ficheiro para qual quer fazer as precisões dento de **projetofinal/data_missing**
ps: Para fazer a precisão é necessário correr treino pelo menos uma vez para ter o modelo primeiro

3. Corra o comando no seu terminal para treinar
```python
python -m projetofinal train --data_path "projetofinal/data_train/01.Dataset FI_06032024.xlsx"
```
ps: Só é preciso correr uma vez, porém se tiver mais dados e quiser atualizar o modelo, correr de novo para treinar novamente com mais dados

4.  Faça a previsão usando o comando abaixo
```python
python -m projetofinal train --order "pred" --format "csv"  
```
ps: indique o formato que o ficheiro dos dados está [csv, xlsx]
