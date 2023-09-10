# remove HTML tags 
#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8

from flask import Flask, jsonify, request

import pickle

# from nltk.tokenize import  word_tokenizen 
from bs4 import BeautifulSoup
import lxml
import html5lib
import tensorflow_hub as hub
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Chargement du modele
with open("usemodel.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/tag_prediction", methods=["POST"])
def predict_tag():
    # Recuperer les donnees de la requete
    data = request.get_json()
    question = data["question"]

    preprocessed_question = preprocess(question)
    embeded_question = embed([preprocessed_question])
    predicted_tags = model.predict([embeded_question])
    # Conversion des tags en liste
    predicted_tags_list = predicted_tags.tolist()

    # Appliquer le seuil de 0.5 pour obtenir des valeurs de 0 ou 1
    df_thresholded = predicted_tags >= 0.5
    targets = pd.read_csv('targets.csv')
    
    target_names = list(targets['target'])
    result = pd.DataFrame(df_thresholded, columns=target_names).T
    prediction = list(result[result[0] == True].index)
    return jsonify({"tags": prediction})

def preprocess(text):
    # Effectuer le pretraitement du texte (ex : suppression des stopwords, normalisation, etc.)
    # Retourner le texte pretraite
    desc_text = clean_text(text)
    word_tokens = tokenizer_fct(desc_text)
#    sw = stop_word_filter_fct(word_tokens)
    lw = lower_start_fct(word_tokens)
    # lw_noun = keep_nouns(lw)
    # lem_w = lemma_fct(lw)    
    transf_desc_text = ' '.join(lw)
    return transf_desc_text
    return text

def tokenizer_fct(sentence) :
    # sentence_clean = re.sub('[;(),\.!?]', '', sentence)
    sentence_clean = sentence.replace('.', ' . ').replace('?', ' ? ').replace('!', ' ! ').replace(';', ' ; ').replace(',', ' , ').replace('[', ' [ ').replace(']', ' ] ').replace(':', ' : ').replace('(', ' ( ').replace(')', ' ) ').replace('+', ' + ').replace('...', ' ... ').replace('\n', ' ')
    word_tokens = word_tokenize(sentence_clean)
    return word_tokens

def clean_text (text):
    soup = BeautifulSoup(text, "html5lib")

    for sentence in soup(['style' , 'script']):
        sentence.decompose()

    return ' '.join(soup.stripped_strings)

def lower_start_fct(list_words) :
    lw = [w.lower() for w in list_words]
    return lw

if __name__ == "__main__":
    app.run(debug=True)

    # une fois les tests realise, enregistrer le modele bagofwords en utilisant pickle, et le charger ici et l'utiliser pour predire les tags