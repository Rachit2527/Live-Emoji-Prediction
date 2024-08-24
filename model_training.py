import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator  
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# Data Preprocessing
train_datagen = ImageDataGenerator(rescale=1./255)  # Normalize to 0-1

# Load your npy files as images (assuming they represent images)
train_generator = train_datagen.flow_from_directory(
    'path/to/training/data',  # Replace with your data path
    target_size=(28, 28),  # Adjust image size if needed
    batch_size=32,
    class_mode='categorical')  # Adjust class mode if needed

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    'path/to/validation/data',
    target_size=(28, 28),
    batch_size=32,
    class_mode='categorical')

# Model Architecture
ip = Input(shape=(28, 28, 3))  # Adjust for your data shape

m = Dense(512, activation="relu")(ip)
m = Dropout(0.2)(m)  # Add dropout for regularization
m = Dense(256, activation="relu")(m)
m = Dropout(0.2)(m)

op = Dense(train_generator.classes.shape[0], activation="softmax")(m)

model = Model(inputs=ip, outputs=op)

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Compile and Train
model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['acc'])
model.fit(train_generator, epochs=50, validation_data=validation_generator, callbacks=[early_stopping])

# Save Model and Labels
model.save("model.h5")
np.save("labels.npy", train_generator.class_indices)  # Save class labels