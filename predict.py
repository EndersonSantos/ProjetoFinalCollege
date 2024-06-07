from flask import Flask, request


from projetofinal.preprocessing import read_pickle
from projetofinal.train_tools import return_embeedings


app = Flask('Decoding')

@app.route('/predict', methods=['POST'])
def predict():
    string = request.get_data(as_text=True)
    (
    inverted_mapping_ter,
    inverted_mapping_sec,
    word2vec_model,
    clf_sec,
    clf_ter,
) = read_pickle('projetofinal/models/xg_boost.pkl')
    
    str_answer = return_embeedings(string,
    word2vec_model,
    clf_ter,
    clf_sec,
    inverted_mapping_ter,
    inverted_mapping_sec,
    'xg_boost')

    result = {
        "Territory": str_answer[0] ,
        "Setor": str_answer[1]
    }
    return result

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)