import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint

# Step 1: Load and preprocess dataset
data_dir = "dataset"
img_size = 64
X, y = [], []
labels = sorted(os.listdir(data_dir))  # sort to maintain label order
label_map = {label: idx for idx, label in enumerate(labels)}

for label in labels:
    folder_path = os.path.join(data_dir, label)
    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (img_size, img_size))
            X.append(img)
            y.append(label_map[label])

X = np.array(X).reshape(-1, img_size, img_size, 1) / 255.0
y = to_categorical(y)

# Step 2: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Build the CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dense(len(labels), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 4: Train and save best model
os.makedirs("model", exist_ok=True)
checkpoint = ModelCheckpoint("model/bestmodel.keras", monitor='val_accuracy', save_best_only=True)

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15, batch_size=16, callbacks=[checkpoint])

print("✅ Model trained and saved to model/bestmodel.keras")
