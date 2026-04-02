import os
import kagglehub
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image


# метод для загрузки данных
def load_images_from_folders(root_path):
    images = []
    labels = []

    class_map = {
        "Brain Tumor": 1,
        "Healthy": 0
    }

    for class_name, label in class_map.items():
        class_path = os.path.join(root_path, class_name)
        for file in os.listdir(class_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                img_path = os.path.join(class_path, file)

                try:
                    img = Image.open(img_path)
                    img = img.convert("RGB")
                    img = img.resize((128, 128))
                    img_array = np.array(img)
                    if img_array.dtype == np.uint16:
                        img_array = img_array.astype(np.float32) / 65535.0
                    else:
                        img_array = img_array.astype(np.float32) / 255.0

                    images.append(img_array)
                    labels.append(label)

                except Exception as e:
                    print(f"Ошибка: {file} -> {e}")

    return np.array(images), np.array(labels)


# метод для проверки сбалансированности классов
def check_class_balance(labels, name):
    unique, counts = np.unique(labels, return_counts=True)

    print(f"\n{name} распределение:")
    for u, c in zip(unique, counts):
        class_name = "Healhty (0)" if u == 0 else "Brain Tumor (1)"
        percent = (c / len(labels)) * 100
        print(f"{class_name}: {c} изображений ({percent:.2f}%)")


# загрузка данных с kaggle
print("Начало загрузки данных")
os.environ["KAGGLEHUB_CACHE"] = "data/kagglehub_cache"

path = kagglehub.dataset_download(
    "preetviradiya/brian-tumor-dataset"
)
print(f"Данные сохранены в {path}")

print("\nПодготовка выборок")
# подготовка выборок
root_path = Path(path) / "Brain Tumor Data Set" / "Brain Tumor Data Set"
X, Y = load_images_from_folders(
    root_path=root_path
)

print("\nРазмеры датасета")
print(f"X: {X.shape}")
print(f"Y: {Y.shape}")

sss = StratifiedShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
train_idx, test_idx = next(sss.split(X, Y))

x_train, x_test = X[train_idx], X[test_idx]
y_train, y_test = Y[train_idx].astype(np.float32), Y[test_idx].astype(np.float32)

print("\nРазмеры выборок")
print(f"x_train: {x_train.shape}")
print(f"x_test: {x_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test: {y_test.shape}")

print("\nСбалансированность классов:")
check_class_balance(y_train, "Train выборка")
check_class_balance(y_test, "Test выборка")

# учёт дисбаланса классов
class_weights = compute_class_weight('balanced',
                                     classes=np.unique(y_train),
                                     y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

# описание модели
model = Sequential([
    Input(shape=(128,128,3)),

    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# сборка модели
model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

print("\nНачало обучения")
# обучение модели
model.fit(x_train, y_train, batch_size=16,
              epochs=5,
              validation_data=(x_test, y_test),
              class_weight=class_weight_dict)

print("\nОбучение завершено")

# проверка модели на тестовых данных
predictions_max = model.predict(x_test)
binary_predictions_max = (predictions_max > 0.5).astype(int).flatten()
accuracy_max = np.mean(y_test == binary_predictions_max)
print(f'\nТочность предсказания на тестовых данных (accuracy) : {accuracy_max * 100:.5f}%')

cm_max = confusion_matrix(y_test, binary_predictions_max)

print("\nМетрики классификации:")
print(classification_report(
    y_test,
    binary_predictions_max,
    target_names=['Healthy', 'Brain Tumor']
))

print("\nСохранение модели")
model.save("model.keras")
print("\nМодель сохранения в model.keras")
