from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from sklearn.model_selection import train_test_split
from data_helpers import load_data


print('Loading data')
x, y, vocabulary, vocabulary_inv = load_data()
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

sequence_length = x.shape[1]  # 56
vocabulary_size = len(vocabulary_inv)  # 18765
embedding_dim = 256
filter_sizes = [3, 4, 5]
num_filters = 512
drop = 0.5

epochs = 100
batch_size = 30

# this returns a tensor
print("Creating Model...")
inputs = keras.Input(shape=(sequence_length,), dtype='int32')
embedding = layers.Embedding(
    input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)(inputs)
reshape = layers.Reshape((sequence_length, embedding_dim, 1))(embedding)

conv_0 = layers.Conv2D(num_filters, kernel_size=(
    filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_1 = layers.Conv2D(num_filters, kernel_size=(
    filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_2 = layers.Conv2D(num_filters, kernel_size=(
    filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

maxpool_0 = layers.MaxPool2D(pool_size=(
    sequence_length - filter_sizes[0] + 1, 1), strides=(1, 1), padding='valid')(conv_0)
maxpool_1 = layers.MaxPool2D(pool_size=(
    sequence_length - filter_sizes[1] + 1, 1), strides=(1, 1), padding='valid')(conv_1)
maxpool_2 = layers.MaxPool2D(pool_size=(
    sequence_length - filter_sizes[2] + 1, 1), strides=(1, 1), padding='valid')(conv_2)

concatenated_tensor = layers.Concatenate(
    axis=1)([maxpool_0, maxpool_1, maxpool_2])
flatten = layers.Flatten()(concatenated_tensor)
dropout = layers.Dropout(drop)(flatten)
output = layers.Dense(units=2, activation='softmax')(dropout)

# this creates a model that includes
model = keras.Model(inputs=inputs, outputs=output)

checkpoint = callbacks.ModelCheckpoint('weights.{epoch:03d}-{val_accuracy:.4f}.hdf5',
                                       monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
adam = optimizers.Adam(learning_rate=1e-4, beta_1=0.9,
                       beta_2=0.999, epsilon=1e-08)

model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
print("Training Model...")
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
          callbacks=[checkpoint], validation_data=(X_test, y_test))  # starts training
