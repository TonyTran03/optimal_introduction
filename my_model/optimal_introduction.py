import pandas as pd
import tensorflow as tf
import numpy as np

from tensorflow import keras
#keras._tf_keras.keras. is the only way I could get the import to work
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences

from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Embedding, LSTM, Dense

data = pd.read_excel('student_profiles_200.xlsx')

# column into list. Note message is the col name that I manually imported
messages = data['message'].dropna().tolist()


#messages = ["I love machine learning", "Deep learning is fun", "NLP is interesting"]
#Sequences: [[1, 2, 3, 4], [5, 4, 6], [7, 6, 8]]
#Max sequence length: 4
#Padded Sequences: 
#[[0 1 2 3 4]
# [0 0 5 4 6]
# [0 0 7 6 8]]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(messages)
sequences = tokenizer.texts_to_sequences(messages)
max_sequence_len = max([len(x) for x in sequences])
sequences = pad_sequences(sequences, maxlen=max_sequence_len, padding='pre') 


#linear stack of 
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=64, input_length=max_sequence_len-1))
model.add(LSTM(100))
model.add(Dense(len(tokenizer.word_index)+1, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Prepare data for training
X = sequences[:,:-1]
y = sequences[:,-1]
y = tf.keras.utils.to_categorical(y, num_classes=len(tokenizer.word_index)+1)

# Train the model
model.fit(X, y, epochs=50)
model.save('myModel.h5')



def generate_message(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=-1)[0]
        output_word = tokenizer.index_word[predicted_word_index]
        seed_text += " " + output_word
    return seed_text

seed_text = "Looking for team"
next_words = 100
generated_message = generate_message(seed_text, next_words, model, max_sequence_len)
print(generated_message)