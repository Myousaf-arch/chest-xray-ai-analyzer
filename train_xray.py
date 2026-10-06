import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, RandomRotation, RandomZoom
from tensorflow.keras.models import Model

# 1. Dataset Paths Setup
base_dir = "dataset"
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

IMG_SIZE = (150, 150)
BATCH_SIZE = 32

print("--- Loading Real Chest X-Ray Dataset ---")
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary',
    shuffle=True
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary',
    shuffle=False
)

# Fix 1: Correct pipeline order (Shuffle -> Cache -> Prefetch)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1000).cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Fix 2: Transfer Learning Architecture (MobileNetV2)
base_model = MobileNetV2(
    input_shape=(150, 150, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze pre-trained weights

inputs = Input(shape=(150, 150, 3))

# Fix 3: Safe Data Augmentation (No horizontal flip for X-Rays)
x = RandomRotation(0.05)(inputs)
x = RandomZoom(0.05)(x)

# MobileNetV2 Specific Preprocessing (-1 to 1 Scaling)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)

# 3. Model Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
)

# Fix 4: Class Weights (Handling Imbalanced Data: ~1341 Normal vs ~3875 Pneumonia)
class_weight = {0: 2.89, 1: 0.67}

# Fix 5: Early Stopping Callback (Overfitting se bachaayega)
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=3, 
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "best_chest_xray_model.keras", 
        save_best_only=True, 
        monitor='val_loss'
    )
]

# 4. Model Training
print("\n--- Model Training Started ---")
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=15,
    class_weight=class_weight,
    callbacks=callbacks
)

# 5. Save Final Model
model.save("chest_xray_model.keras")
print("\n--- Model Saved Successfully as 'chest_xray_model.keras' ---")