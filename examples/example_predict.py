from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
from sentiment import data_helpers

trained_model = load_model("weights.001-0.8696.hdf5")
adam = optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
trained_model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

x = data_helpers.prepare_sentence("This is the greatest movie of all time")
trained_model.predict(x)