# ProjetoFinalCollege
1. Instale todas as dependencias para o projeto: 
```
pip install -r requirements.txt
```
ps: Certifique que está no diretório certo

2. Adicione os dados necessários
- Para treinar o modelo coloque dentro do folder **projetofinal/data_train** adicione os dados que utilizará para treinar o modelo
- Adicione o ficheiro para qual quer fazer as precisões dento de **projetofinal/data_missing**
- Também adicione um data set para avaliar o modelo
ps: Para fazer a precisão é necessário correr treino pelo menos uma vez para ter o modelo primeiro

3. Corra o comando no seu terminal para treinar
```python
python -m projetofinal train --data_path "projetofinal/data_train/data_path"
```
ps: 
- Só é preciso correr uma vez, porém se tiver mais dados e quiser atualizar o modelo, correr de novo para treinar novamente com mais dados
- Mude o data_path para onde os dados estão no seu computador

4.  Faça a previsão usando o comando abaixo
```python
python -m projetofinal train --order "pred" --format "csv"  
```
ps: indique o formato que o ficheiro dos dados está [csv, xlsx]

5. Para saber a performance do seu modelo compre
```python
python -m projetofinal train --order "eval" --data_path "projetofinal/data_train/data_path"  --format "csv"
```
- Mude o data_path para onde os dados estão no seu computador
- Mude o fomato para o formato do seu  ficheiro .csv ou .xlsx