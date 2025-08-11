# # app.py

# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from sklearn.preprocessing import LabelEncoder
# from fastapi import FastAPI
# from pydantic import BaseModel

# # Initialize FastAPI app
# app = FastAPI(title="Text Classification Spam Detector")

# # Load and preprocess dataset
# df = pd.read_csv("data/spam.csv", encoding='latin-1')[['v1', 'v2']]
# df.columns = ['label', 'text']

# # Encode labels (ham=0, spam=1)
# le = LabelEncoder()
# df['label'] = le.fit_transform(df['label'])

# # Split dataset
# X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2)

# # Tokenization and Padding
# vocab_size = 5000
# max_length = 100
# oov_tok = "<OOV>"

# tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
# tokenizer.fit_on_texts(X_train)

# X_train_seq = tokenizer.texts_to_sequences(X_train)
# X_test_seq = tokenizer.texts_to_sequences(X_test)

# X_train_pad = pad_sequences(X_train_seq, maxlen=max_length)
# X_test_pad = pad_sequences(X_test_seq, maxlen=max_length)

# # Build model
# model = tf.keras.Sequential([
#     tf.keras.layers.Embedding(vocab_size, 16, input_length=max_length),
#     tf.keras.layers.GlobalAveragePooling1D(),
#     tf.keras.layers.Dense(24, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# # Train model
# model.fit(X_train_pad, y_train, epochs=10, validation_data=(X_test_pad, y_test), batch_size=32)

# # Save model
# model.save("model/text_classifier_model.h5")

# # Reload model and tokenizer for prediction
# loaded_model = tf.keras.models.load_model("model/text_classifier_model.h5")

# # Request model
# class TextRequest(BaseModel):
#     text: str

# @app.get("/")
# def root():
#     return {"message": "Text Classifier API is running ✅"}

# @app.post("/predict/")
# def predict_text(req: TextRequest):
#     text = req.text
#     seq = tokenizer.texts_to_sequences([text])
#     padded = pad_sequences(seq, maxlen=max_length)
#     prediction = loaded_model.predict(padded)[0][0]

#     return {
#         "input_text": text,
#         "prediction": "spam" if prediction > 0.5 else "ham",
#         "confidence": float(prediction)
#     }
# app.py

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from fastapi import FastAPI
from pydantic import BaseModel

# ✅ Print TensorFlow version
print("TensorFlow version:", tf.__version__)

# Initialize FastAPI app
app = FastAPI(title="Text Classification Spam Detector")

# Load and preprocess dataset
df = pd.read_csv("data/spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']

# Encode labels (ham=0, spam=1)
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2)

# Tokenization and Padding
vocab_size = 5000
max_length = 100
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_length)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 16, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X_train_pad, y_train, epochs=10, validation_data=(X_test_pad, y_test), batch_size=32)

# Save model
model.save("model/text_classifier_model.h5")

# Reload model and tokenizer for prediction
loaded_model = tf.keras.models.load_model("model/text_classifier_model.h5")

# Request model
class TextRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Text Classifier API is running ✅"}

@app.post("/predict/")
def predict_text(req: TextRequest):
    text = req.text
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_length)
    prediction = loaded_model.predict(padded)[0][0]

    return {
        "input_text": text,
        "prediction": "spam" if prediction > 0.5 else "ham",
        "confidence": float(prediction)
    }
