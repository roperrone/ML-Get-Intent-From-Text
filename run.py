# -*- coding: utf-8 -*-

import spacy
import numpy as np
import math  
from joblib import dump,load
from flask import Flask, json, request
import re
import unidecode
import urllib.request

api = Flask(__name__)

@api.route('/', methods=['GET'])
def get_intention():
  if 'q' in request.args:
     query = request.args.get('q')
     
     response = urllib.request.urlopen("https://2jfhg21asd.execute-api.eu-west-1.amazonaws.com/dev/app/getSupportedPlants")
     data = json.loads(response.read())

     plants = []
     for plant in data:
        plants.append(plant["species"])

     intention = getIntention(query)
     plant = identifyWantedPlant(plants, query) 
     return json.dumps([{"response": "success", "results": intention, "plant": plant}])
  else:
     return json.dumps([{"response": "error"}])

def getIntention(sentence):
    
    #load trained model
    clf_svm = load('clf_svm.joblib')
    
    # Pre-processing
    sentence = sentence.lower()

    regex = re.compile("plante([^r]|$)")
    sentence = regex.sub('', sentence)

    tokens = nlp_fr(sentence)

    words = []
    vectorizer = load('vectorizer.joblib')
                
    # Lemmatize
    for token in tokens:
        if (token.lemma_ != 'plante' and token.lemma_ != 't'): 
            words.append(str(token.lemma_))

    j = 0;
    vector = vectorizer.get_feature_names()

    # Create vector
    for word in vector:
        vector[j] = words.count(word);           
        j += 1


    p = clf_svm.predict([vector])
    
    intentions = load('intentions.joblib')
	
    score = clf_svm.predict_proba([vector])
    best = score[0][int(p[0])]

    score = np.delete(score, np.where(score == best))
    second_best = np.amax(score)
    intention_score = 1/(1+math.exp(-(-0.3+(best-second_best))*10))

    return [intentions[int(p[0])], intention_score]

def identifyWantedPlant(plant_list, sentence):
    # plant_list composed of the species and nicknames of the possessed plants
    expr = unidecode.unidecode(sentence.lower())
    
    for count,name in enumerate(plant_list):
        if re.findall(unidecode.unidecode(name.lower()), expr):
            # if found, returns the in the given list
            return name
    return

if __name__ == "__main__":
   nlp_fr = spacy.load('fr_core_news_sm')
   api.run(host='0.0.0.0', port=5000)
