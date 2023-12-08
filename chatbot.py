import json
import numpy as np
import pickle
import random
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
with open('intents.json') as file:
    intents = json.load(file)

words = pickle.load(open('word.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

model = load_model('chatbot.model',)

def clean_sentence(sentence):
    sentence_word = nltk.word_tokenize(sentence)
    sentence_word = [lemmatizer.lemmatize(word) for word in sentence_word]
    return sentence_word

def bag_of_words(sentence):
    sentence_word = clean_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_word:
        for i , word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results = sorted(results, key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intent = intents_json['intents']
    for i in list_of_intent:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print("Bot Running")
while True:
    message = input('')
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)
