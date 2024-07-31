import pandas as pd
import tensorflow as tf
from tensorflow import keras

#keras._tf_keras.keras. is the only way I could get the import to work
from keras._tf_keras.keras.preprocessing.text import Tokenizer


from keras._tf_keras.keras.preprocessing.sequence import pad_sequences

from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Embedding, LSTM, Dense

data = pd.read_csv('sample_text_with_emote.xlsx')

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