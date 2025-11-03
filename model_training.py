# model_training.py

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os

# --- Configuration ---
DATA_PATH = 'fer2013.csv'  # IMPORTANT: Update this path to your dataset
MODEL_PATH = 'face_emotionModel.h5'
IMG_WIDTH = 48
IMG_HEIGHT = 48
NUM_CLASSES = 7
EMOTION_LABELS = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}


# --- 1. Load and Preprocess Data ---
def load_and_preprocess_data(data_path):
    print(f"Loading data from {data_path}...")
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {data_path}")
        print("Please download the FER2013 dataset and update the DATA_PATH variable.")
        return None, None

    pixels = data['pixels'].tolist()
    faces = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(IMG_WIDTH, IMG_HEIGHT)
        faces.append(face.astype('float32'))

    faces = np.asarray(faces)
    # Add a channel dimension (for grayscale)
    faces = np.expand_dims(faces, -1)

    # Normalize data
    faces /= 255.0

    # Get labels and one-hot encode them
    emotions = to_categorical(data['emotion'], NUM_CLASSES)

    print(f"Data loaded. Shapes: Faces={faces.shape}, Emotions={emotions.shape}")
    return faces, emotions


# --- 2. Build the CNN Model ---
def build_model():
    print("Building model architecture...")
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same', input_shape=(IMG_WIDTH, IMG_HEIGHT, 1), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (5, 5), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(NUM_CLASSES, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.0001),
                  metrics=['accuracy'])

    model.summary()
    return model


# --- Main execution ---
if __name__ == "__main__":
    X, y = load_and_preprocess_data(DATA_PATH)

    if X is not None:
        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

        model = build_model()

        print("Starting model training...")
        # Train the model
        model.fit(
            X_train, y_train,
            batch_size=64,
            epochs=50,  # You can increase this for better accuracy, but it will take longer
            validation_data=(X_val, y_val),
            verbose=1
        )

        # Save the trained model
        model.save(MODEL_PATH)
        print(f"\nTraining complete. Model saved to {MODEL_PATH}")