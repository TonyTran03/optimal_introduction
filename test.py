import pandas as pd

import tensorflow as tf
import numpy as np

from tensorflow import keras
#keras._tf_keras.keras. is the only way I could get the import to work
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences

from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Embedding, LSTM, Dense, Dropout
from keras._tf_keras.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Load data
data = pd.read_excel('student_profiles_200.xlsx')

# Inspect data balance
messages = data['message'].dropna().tolist()
tokenizer = Tokenizer()
tokenizer.fit_on_texts(messages)
word_counts = tokenizer.word_counts

# Check the most common words
common_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
print("Most common words and their counts:", common_words[:20])
