import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, RandomFlip, RandomRotation, RandomZoom, Rescaling

print("TensorFlow Version:", tf.__version__)

# 1. Dummy Data Generate Karna (100 images, 150x150 pixels, 3 channels)
X_train = np.random.randint(0, 256, size=(100, 150, 150, 3)).astype(np.float32)
y_train = np.random.randint(0, 2, size=(100, 1))

# 2. Modern Keras 3 CNN Model Architecture with Data Augmentation
model = Sequential([
    # Input Layer (Warning Fix karne ke liye)
    Input(shape=(150, 150, 3)),
    
    # Preprocessing: Pixel values ko 0-255 se 0-1 scale par lana
    Rescaling(1./255),
    
    # Data Augmentation (Overfitting se bachane aur variety barhane ke liye)
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.1),
    
    # Feature Extraction (CNN Layers)
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    # Flattening & Classification
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid') # Binary Output (0/1)
])

# 3. Model Compile Karna
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 4. Model Summary Print
model.summary()

# 5. Fast Test Run
print("\n--- Training Model with Keras 3 Pipeline ---")
model.fit(X_train, y_train, epochs=5, batch_size=16)
print("\n--- Modern Pipeline Ready Successfully! ---")