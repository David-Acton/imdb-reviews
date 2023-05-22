#Train

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# Step 3: Prepare the Data
good_reviews = "good_reviews.txt"  # Path to the file containing good text
bad_reviews = "bad_reviews.txt"    # Path to the file containing bad text

# Load the text data
with open(good_reviews, "r") as file:
    good_text = file.read().splitlines()
    
with open(bad_reviews, "r") as file:
    bad_text = file.read().splitlines()

texts = good_text + bad_text
labels = [1] * len(good_text) + [0] * len(bad_text)

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

max_sequence_length = max(len(seq) for seq in sequences)
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_sequence_length)

labels = np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100 

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Trains the Model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test loss: {loss:.4f}")
print(f"Test accuracy: {accuracy:.4f}")

model.save("my_model.h5")  



#Predict

import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("my_model.h5")

def preprocess_sentence(sentence):

    sequence = tokenizer.texts_to_sequences([sentence])

    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_sequence_length)
    
    return padded_sequence

def predict_sentiment(sentence):
    preprocessed_sentence = preprocess_sentence(sentence)
    
    predictions = model.predict(preprocessed_sentence)
    
    sentiment_label = "Good" if predictions[0] >= 0.5 else "Bad"
    
    return sentiment_label

# Test a sentence
test_sentence = "First, let's address the elephant in the room. This movie is getting review bombed by a certain crowd for the progressive ideals of the actress playing Captain Marvel. They not only review bomb by pressing the star rating, but also by posting seemingly well-thought out reviews giving the movie 1 star, 2 stars, 3 stars, anything to make the overall rating go down.Dont believe the hate, this movie is EVERY BIT as much fun as other Marvel movies. Humor? Check. Well acted? Check. Lovely references? Check. Action-packed? Check. Heartfelt? Check.I wouldnt place it at the very top, but its a solid 8 when matched against others.Going into more detail would mean spoilers and I believe that people should be able to see beyond the hatred coming from fragile egos. Just take it from this fellow fan of the MCU: Captain Marvel is just as magical, just as funny, just as worthy as any other title of this franchise. Go see it!"
predicted_sentiment = predict_sentiment(test_sentence)
print(f"Sentence: {test_sentence}")
print(f"Predicted Sentiment: {predicted_sentiment}")