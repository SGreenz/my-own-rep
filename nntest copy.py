import os
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
import cv2
import random
from keras.models import Model
from keras.layers import Input, Conv2D, Maxpooling2D, UpSampling2D, Concatenate

from skimage import measure
from skimage.io import imread, imsave, imshow
from skimage.transform import resize
from skimage.filters import gaussian
from skimage.morphology import dilation, disk
from skimage.draw import polygon, polygon_perimeter

CLASSES = 2     # Количество классов
SAMPLE_SIZE = (256, 256)    # Размер изображения подаваемого на вход сети
OUTPUT_SIZE = (656, 1184)   # Размер изображения получаемого на выходе
BATCH_SIZE = 32  # Размер батча

# Путь к папке с изображениями и масками
data_dir = '/home/egorov_pa@rodina.loc/PROJECTS/NNthread/program'

# Загрузка изображений и масок, выделение по каналам классов объектов в масках


def load_data(image, mask):
    # Загрузка изображений
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image)
    image = tf.image.resize(image, OUTPUT_SIZE)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = image / 255.0
    # Загрузка масок
    mask = tf.io.read_file(mask)
    mask = tf.io.decode_png(mask)
   # mask = tf.image.rgb_to_grayscale(mask)  # Форматирование RGB в оттенки серого
    mask = tf.image.resize(mask, OUTPUT_SIZE)
    mask = tf.image.convert_image_dtype(mask, tf.float32)

    masks = []

    for i in range(CLASSES):
        masks.append(tf.where(tf.equal(mask, float(i)), 1.0, 0.0))

    masks = tf.stack(masks, axis=2)
    masks = tf.reshape(masks, OUTPUT_SIZE + (CLASSES,))

    return image, masks


# Разделение данных на тренировочные и тестовые
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Получение списка файлов изображений и масок
image_files = os.listdir(os.path.join(data_dir, 'images_try'))
mask_files = os.listdir(os.path.join(data_dir, 'masks_try'))

# Разбиение на тренировочные, валидационные и тестовые данные
num_images = len(image_files)
num_train = int(num_images * train_ratio)
num_val = int(num_images * val_ratio)
num_test = int(num_images * test_ratio)

train_indexes = random.sample(range(num_images), num_train)
val_indexes = random.sample(
    set(range(num_images)) - set(train_indexes), num_val)
test_indexes = set(range(num_images)) - set(train_indexes) - set(val_indexes)

train_image_files = [image_files[i] for i in train_indexes]
train_mask_files = [mask_files[i] for i in train_indexes]
val_image_files = [image_files[i] for i in val_indexes]
val_mask_files = [mask_files[i] for i in train_indexes]
test_image_files = [image_files[i] for i in test_indexes]
test_mask_files = [mask_files[i] for i in train_indexes]

# Cоздание генератора данных


def data_generator(image_files, mask_files, batch_size):
    while True:
        indexes = random.sample(range(len(image_files)), batch_size)
        batch_image_files = [image_files[i] for i in indexes]
        batch_mask_files = [mask_files[i] for i in indexes]
        batch_images = []
        batch_masks = []
        for image_file, mask_file in zip(batch_image_files, batch_mask_files):
            image, mask = load_data(image_file, mask_file)
            batch_images.append(image)
            batch_masks.append(mask)
        yield np.array(batch_images), np.array(batch_masks)


# Создание генераторов для тренировочных, валидационных и тестовых данных
train_generator = data_generator(
    train_image_files, train_mask_files, batch_size=BATCH_SIZE)
val_generator = data_generator(
    val_image_files, val_mask_files, batch_size=BATCH_SIZE)
test_generator = data_generator(
    test_image_files, test_mask_files, batch_size=BATCH_SIZE)

# Создание блока инициализации


def initial_block(input_layer, nb_filter):
    # Layers for the initial block of the model
    convolution = Conv2D(nb_filter, (3, 3), padding='same')(input_layer)
    out = Conv2D(nb_filter, (3, 3), padding='same')(convolution)
    max_pool = MaxPooling2D(2)(out)
    return max_pool

# Создание блока "бутылочное горлышко"


def bottleneck(input_layer, nb_filter, output_size, dilation_rate):
    # Layers for each bottleneck block
    convolution = Conv2D(
        nb_filter, (1, 1), dilation_rate=dilation_rate)(input_layer)
    out = Conv2D(nb_filter, (3, 3), padding='same',
                 dilation_rate=dilation_rate, activation='relu')(convolution)
    out = Conv2D(nb_filter * output_size, (1, 1))(out)
    out = MaxPooling2D(2)(out)
    return out

# Создание конечного блока


def final_block(input_layer, nb_filter):
    # Layers for the final block of the model
    convolution = Conv2D(nb_filter, (3, 3), padding='same')(input_layer)
    out = Conv2D(nb_filter, (3, 3), padding='same')(convolution)
    return out

# Функция создания модели


def create_enet_model(input_shape, num_classes):
    # Определение архитектуры модели ENet
    input_layer = Input(shape=input_shape + (3,))

    # Encoder
    conv1 = initial_block(input_layer, nb_filter=32)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    enet_block1_0 = bottleneck(
        pool1, nb_filter=64, output_size=4, dilation_rate=(1, 1))
    enet_block1_1 = bottleneck(
        enet_block1_0, nb_filter=64, output_size=4, dilation_rate=(1, 1))
    enet_block1_2 = bottleneck(
        enet_block1_1, nb_filter=64, output_size=4, dilation_rate=(1, 1))
    enet_block1_3 = bottleneck(
        enet_block1_2, nb_filter=64, output_size=4, dilation_rate=(2, 2))
    enet_block1_4 = bottleneck(
        enet_block1_3, nb_filter=64, output_size=4, dilation_rate=(4, 4))
    enet_block1_5 = bottleneck(
        enet_block1_4, nb_filter=64, output_size=4, dilation_rate=(8, 8))
    enet_block1_6 = bottleneck(
        enet_block1_5, nb_filter=64, output_size=4, dilation_rate=(16, 16))

    # Decoder
    upconv1 = UpSampling2D(size=(2, 2))(enet_block1_6)
    concat1 = Concatenate([upconv1, enet_block1_2], axis=-1)
    conv2 = bottleneck(concat1, nb_filter=64,
                       output_size=4, dilation_rate=(1, 1))

    upconv2 = UpSampling2D(size=(2, 2))(conv2)
    concat2 = Concatenate([upconv2, conv1], axis=-1)
    conv3 = final_block(concat2, nb_filter=num_classes)

    # Создание модели
    model = Model(inputs=input_layer, outputs=conv3)
    return model


# Выполенине функции создания модели
enet_like = create_enet_model(SAMPLE_SIZE, CLASSES)


enet_like.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

history_dice = enet_like.fit(train_generator, steps_per_epoch=num_train//BATCH_SIZE, epochs=10,
                             validation_data=val_generator, validation_steps=num_val//BATCH_SIZE)

enet_like.save_weights(
    '/home/egorov_pa@RODINA.LOC/PROJECTS/NNthread/ENet_model/enet_like')

enet_like.evaluate(test_generator, steps=num_test//BATCH_SIZE)

# unet_like.load_weights('/home/egorov_pa@RODINA.LOC/PROJECTS/NNthread/program/networks/unet_like')

# rgb_colors = [
#    (0, 0, 0),
#    (255, 0, 0),
#    (0, 255, 0),
#    (255, 255, 0)
# ]

# frames = sorted(glob.glob('/home/egorov_pa@RODINA.LOC/PROJECTS/NNthread/program/images/*.jpg'))

# for filename in frames:
#    frame = imread(filename)
#    sample = resize(frame, SAMPLE_SIZE)

#    predict = unet_like.predict(sample.reshape((1,) + SAMPLE_SIZE + (3,)))
#    predict = predict.reshape(SAMPLE_SIZE + (CLASSES,))

#    scale = frame.shape[0] / SAMPLE_SIZE[0], frame.shape[1] / SAMPLE_SIZE[1]

#    frame = (frame / 1.5).astype(np.uint8)

#    for channel in range(1, CLASSES):
#        contour_overlay = np.zeros((frame.shape[0], frame.shape[1]))
#        contours = measure.find_contours(np.array(predict[:, :, channel]))

#        try:
#            for contour in contours:
#                rr, cc = polygon_perimeter(contour[:, 0] * scale[0],
#                                           contour[:, 1] * scale[1],
#                                           shape=contour_overlay.shape)

#                contour_overlay[rr, cc] = 1

#            contour_overlay = dilation(contour_overlay, disk(1))
#            frame[contour_overlay == 1] = rgb_colors[channel]
#        except:
#            pass

#    imsave(f'/home/egorov_pa@RODINA.LOC/PROJECTS/NNthread/program/res/{os.path.basename(filename)}', frame)
