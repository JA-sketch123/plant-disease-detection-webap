import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# ---------------- SETTINGS ----------------
IMG_SIZE = 300        # Reduced from 300 → faster training
BATCH_SIZE = 16       # Reduced from 32 → less RAM usage
EPOCHS = 8       # Balanced training

DATASET_PATH = "dataset"

# ---------------- DATA AUGMENTATION ----------------
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=25,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset="training",
    class_mode="categorical"
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset="validation",
    class_mode="categorical"
)

# ---------------- LOAD BASE MODEL ----------------
base_model = EfficientNetB3(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze most layers
for layer in base_model.layers:
    layer.trainable = False

# Unfreeze last few layers (fine-tuning)
for layer in base_model.layers[-80:]:
    layer.trainable = True

# ---------------- CUSTOM CLASSIFIER ----------------
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.4)(x)

output = Dense(train_data.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# ---------------- COMPILE MODEL ----------------
model.compile(
    optimizer=Adam(learning_rate=0.0005),   # lower LR for fine-tuning
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ---------------- CALLBACKS ----------------
early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "plant_disease_model.h5",
    monitor="val_accuracy",
    save_best_only=True
)

# ---------------- TRAIN ----------------
print("\n========== TRAINING STARTED ==========\n")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint]
)

# ---------------- SAVE MODEL ----------------
model.save("plant_disease_model.h5")

print("\n✅ Model saved successfully!")
print("\n🎯 Training Complete!")