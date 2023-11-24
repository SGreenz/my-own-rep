import os
import numpy as np
import cv2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from enet import ENet  # импортируем модель ENet из файла enet.py

# функция для загрузки списка файлов из папки


def load_file_list(folder_path):
    file_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            file_list.append(os.path.join(folder_path, filename))
    return file_list

# функция для загрузки изображения и соответствующей маски из файлов на диске


def load_image_and_mask(image_path, mask_folder):
    mask_path = os.path.join(mask_folder, os.path.splitext(
        os.path.basename(image_path))[0] + '.png')
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    image = preprocess_image(image)
    mask = preprocess_mask(mask)
    return image, mask

# функция для предварительной обработки изображения


def preprocess_image(image):
    # применяем необходимые преобразования
    return image

# функция для предварительной обработки маски


def preprocess_mask(mask):
    # применяем необходимые преобразования
    return mask

# функция для создания батч генератора


def batch_generator(image_paths, mask_folder, batch_size):
    i = 0
    while True:
        batch_images = []
        batch_masks = []
        for j in range(batch_size):
            if i == len(image_paths):
                i = 0
            image_path = image_paths[i]
            image, mask = load_image_and_mask(image_path, mask_folder)
            batch_images.append(image)
            batch_masks.append(mask)
            i += 1
        yield np.array(batch_images), np.array(batch_masks)


# путь к папке с изображениями
image_folder = 'path/to/image/folder'
# путь к папке с масками
mask_folder = 'path/to/mask/folder'
# список файлов изображений
image_paths = load_file_list(image_folder)

# создаем батч генератор
batch_size = 8
train_generator = batch_generator(image_paths, mask_folder, batch_size)

# создаем модель ENet
input_shape = (512, 512, 3)  # размер входного изображения
num_classes = 2  # количество классов (фон и объект)
enet = ENet(input_shape, num_classes)

# компилируем модель
optimizer = Adam(lr=1e-4)
loss = 'sparse_categorical_crossentropy'
metrics = ['accuracy']
enet.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# создаем коллбэки для сохранения модели и остановки обучения
model_checkpoint = ModelCheckpoint(
    'enet_model.h5', save_best_only=True, save_weights_only=True)
early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

# обучаем модель
epochs = 50
steps_per_epoch = len(image_paths) // batch_size
history = enet.fit(train_generator, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=[
                   model_checkpoint, early_stopping])
