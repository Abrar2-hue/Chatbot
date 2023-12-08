import json
import numpy as np
import pickle
import random
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay

lemmatizer = WordNetLemmatizer()
with open('intents.json') as file:
    intents = json.load(file)

words = []
documents = []
classes = []
ignore_letters = ['.', '!', '?', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open('word.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Machine Learning Part
training = []
output_empty = [0] * len(classes)
for document in documents:
    bag = []
    word_pattern = document[0]
    word_pattern = [lemmatizer.lemmatize(word.lower()) for word in word_pattern]
    for word in words:
        bag.append(1) if word in word_pattern else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)

train_x = np.array([bag for bag, _ in training])  # Extracting bag-of-words as input
train_y = np.array([output_row for _, output_row in training])  # Extracting output rows as labels

# Build Neural Network
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Create a learning rate schedule with decay
initial_learning_rate = 0.01
decay_steps = 1000
decay_rate = 0.96
learning_rate_fn = ExponentialDecay(initial_learning_rate, decay_steps, decay_rate)
sgd = SGD(learning_rate=learning_rate_fn, momentum=0.09, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
model.save('chatbot.model',hist)
print('Done')
