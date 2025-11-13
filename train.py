import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ========================
# Paths
# ========================
train_dir = r'dataset/train'
valid_dir = r'dataset/valid'
test_dir = r'dataset/test'

# Safety check for dataset paths
print("Checking dataset paths...")
print("Train path exists:", os.path.exists(train_dir))
print("Valid path exists:", os.path.exists(valid_dir))
print("Test path exists:", os.path.exists(test_dir))

if not all([os.path.exists(train_dir), os.path.exists(valid_dir), os.path.exists(test_dir)]):
    raise FileNotFoundError("❌ One or more dataset folders not found. Check your paths!")

# ========================
# Data Generators
# ========================
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20,
                                   width_shift_range=0.2, height_shift_range=0.2,
                                   zoom_range=0.2, shear_range=0.2, horizontal_flip=True)

valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

valid_gen = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# ========================
# Model: Transfer Learning (VGG16)
# ========================
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom classifier
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ========================
# Callbacks
# ========================
checkpoint = ModelCheckpoint("models/best_vgg16_model.h5",
                             monitor="val_accuracy",
                             save_best_only=True,
                             verbose=1)

early_stop = EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)

lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, verbose=1)

# ========================
# Training
# ========================
history = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=30,
    callbacks=[checkpoint, early_stop, lr_scheduler]
)

# ========================
# Evaluation
# ========================
print("\nEvaluating on test set...")
loss, acc = model.evaluate(test_gen)
print(f"✅ Test Accuracy: {acc*100:.2f}%")

# ========================
# Plot Training History
# ========================
plt.figure(figsize=(12,5))

# Accuracy plot
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss plot
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ========================
# Detailed Evaluation
# ========================
y_true = test_gen.classes
y_pred = model.predict(test_gen)
y_pred_classes = np.argmax(y_pred, axis=1)

# Class labels
class_labels = list(test_gen.class_indices.keys())

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=class_labels))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
 