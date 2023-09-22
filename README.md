# SentimentAnalysisFinancialData
I apply an LSTM model for sentiment analysis over a financial review data. Data has two columns: "Review' and 'Sentiment'. I use a tokenizer for the review data and use get_dummies to convert sentiment classes (negative, neutral, positive) to one hot encodings. 
```

```python
# sentiment analysis
# 1. get a source for the data, and read it
# 2. the source is Kaggle (https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis?resource=download)
import pandas as pd
import numpy as np
data = pd.read_csv("FinanceSentimentData.csv")

# OVERALL ALGORITHM
# 1. convert col1 to a list of sentences, then clean the list of sentences
# 2. Use Tokenizer to create vectors of tokens (integers) in place of words
# 3. Process the output and make them each a vector of size two (positive and negative)
inputData = list(data[:]['Sentence'])

# Now clean the inputData: a. replace 's by blank, b. keep only alphabet, c. make lower
import re
inputData = [re.sub("'s ", " ", sent) for sent in inputData]
inputData = [re.sub("[^a-zA-Z]", ' ', sent) for sent in inputData]
inputData = [re.sub("   ", ' ', sent) for sent in inputData]
inputData = [re.sub("  ", ' ', sent) for sent in inputData]
inputData = [re.sub(" [a-zA-Z] ", ' ', sent) for sent in inputData]
inputData = [sent.lower() for sent in inputData]

```python
# use Tokenizer to tokenize the sentences
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()

# first thing to do is to fit_on_texts
#tokenizer.fit_on_texts(inputData+list(set(list(data['Sentiment']))))
tokenizer.fit_on_texts(inputData)

# now convert inputData sentences to sequences
sequences = tokenizer.texts_to_sequences(pd.Series(inputData))

# find the maxLen among all sequences; then pre-pad other sequences accordingly
# Make X ready
maxLen = max(len(seq) for seq in sequences)
from tensorflow.keras.preprocessing.sequence import pad_sequences
X = pad_sequences(sequences, maxLen, truncating='pre')

# Make y ready: y is created from data['Sentiment']
y = np.array(pd.get_dummies(list(data['Sentiment'])))

# create train and test sets
from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
```

```python
# create LSTM model
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

model = Sequential()
model.add(Embedding(len(set(tokenizer.word_index.keys()))+1, 60, input_length = maxLen, trainable=True))
model.add(LSTM(150, recurrent_dropout=0.1, dropout=0.1, ))
model.add(Dense(len(set(list(data['Sentiment']))), 'softmax'))
#print(model.summary())

# compile and train
model.compile(loss='categorical_crossentropy', metrics='acc', optimizer='adam')
model.fit(X_tr, y_tr, epochs=10, verbose=1, validation_data=(X_val, y_val))
```

```python
# predict
seed_text = "us sanctions put gazprom shell alliance plans in jeopardy"
# first convert it to sequence using texts_to_sequences
# 2nd apply pad_sequences
# 3rd use model.predict: from the output probabilities find the max, get its index, get its word, print
seed_tokens = tokenizer.texts_to_sequences([seed_text])
seed_tokens = pad_sequences(seed_tokens, maxLen, truncating='pre')
yhat_probs = model.predict(seed_tokens)
outputWords = sorted(list(data['Sentiment']))
outputWords[np.argmax(yhat_probs)]
```
