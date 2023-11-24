import os
import glob
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

# Функция для загрузки изображений и масок


def load_data(image_path, mask_path):
    def preprocess_image(image):
        image = tf.image.decode_jpeg(tf.io.read_file(image_path), channels=3)
        image = tf.image.resize(image, (256, 256))
        image = image / 255.0
        return image

    def preprocess_mask(mask):
        mask = tf.image.decode_png(tf.io.read_file(mask_path), channels=1)
        mask = tf.image.resize(mask, (256, 256))
        mask = mask / 255.0
        return mask

    image = tf.py_function(preprocess_image, [image_path], tf.float32)
    mask = tf.py_function(preprocess_mask, [mask_path], tf.float32)

    return image, mask


# Путь к папке с изображениями и масками
data_dir = "/путь/к/папке/с/датасетом"

# Список файлов изображений и масок
image_files = glob.glob(data_dir + "/images/*.jpg")
mask_files = glob.glob(data_dir + "/masks/*.png")

# Функция для разделения данных на тренировочные и тестовые наборы


def split_data(image_files, mask_files, split_ratio=0.8):
    # Случайное перемешивание индексов
    indices = list(range(len(image_files)))
    random.shuffle(indices)

    # Определение количества тренировочных и тестовых данных
    split_index = int(len(indices) * split_ratio)

    # Разделение данных
    train_image_files = [image_files[i] for i in indices[:split_index]]
    train_mask_files = [mask_files[i] for i in indices[:split_index]]
    test_image_files = [image_files[i] for i in indices[split_index:]]
    test_mask_files = [mask_files[i] for i in indices[split_index:]]

    return train_image_files, train_mask_files, test_image_files, test_mask_files


# Разделение данных на тренировочные и тестовые наборы
train_image_files, train_mask_files, test_image_files, test_mask_files = split_data(
    image_files, mask_files)

# Загрузка и предобработка тренировочных данных


def load_train_data(image_files, mask_files):
    image_data = []
    mask_data = []
    for image_file, mask_file in zip(image_files, mask_files):
        image, mask = load_data(image_file, mask_file)
        image_data.append(image)
        mask_data.append(mask)
    return image_data, mask_data

# Загрузка и предобработка тестовых данных


def load_test_data(image_files, mask_files):
    image_data = []
    mask_data = []
    for image_file, mask_file in zip(image_files, mask_files):
        image, mask = load_data(image_file, mask_file)
        image_data.append(image)
        mask_data.append(mask)
    return image_data, mask_data


# Загрузка и предобработка тренировочных данных
train_images, train_masks = load_train_data(
    train_image_files, train_mask_files)

# Загрузка и предобработка тестовых данных
test_images, test_masks = load_test_data(test_image_files, test_mask_files)

# Создание модели ENet
model = create_enet_model()

# Компиляция модели
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='binary_crossentropy', metrics=['accuracy'])

# Создание объектов Dataset для тренировочных и тестовых данных
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_masks))
train_dataset = train_dataset.shuffle(len(train_images)).batch(batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_masks))
test_dataset = test_dataset.batch(batch_size)

# Обучение модели
model.fit(train_dataset, epochs=10)

# Оценка модели
model.evaluate(test_dataset)
