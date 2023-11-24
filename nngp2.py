import os
import glob
import random
import numpy as np
from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate

def load_data(image_path, mask_path):
    image = Image.open(image_path)
    mask = Image.open(mask_path)
    return np.array(image), np.array(mask)

def create_enet_model(input_shape, num_classes):
    # Определение архитектуры модели ENet
    input_layer = Input(shape=input_shape)

    # Encoder
    conv1 = initial_block(input_layer, nb_filter=13)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    enet_block1_0 = bottleneck(pool1, nb_filter=64, output_size=4, dilation_rate=(1, 1))
    enet_block1_1 = bottleneck(enet_block1_0, nb_filter=64, output_size=4, dilation_rate=(1, 1))
    enet_block1_2 = bottleneck(enet_block1_1, nb_filter=64, output_size=4, dilation_rate=(1, 1))
    enet_block1_3 = bottleneck(enet_block1_2, nb_filter=64, output_size=4, dilation_rate=(2, 2))
    enet_block1_4 = bottleneck(enet_block1_3, nb_filter=64, output_size=4, dilation_rate=(4, 4))
    enet_block1_5 = bottleneck(enet_block1_4, nb_filter=64, output_size=4, dilation_rate=(8, 8))
    enet_block1_6 = bottleneck(enet_block1_5, nb_filter=64, output_size=4, dilation_rate=(16, 16))

    # Decoder
    upconv1 = UpSampling2D(size=(2, 2))(enet_block1_6)
    concat1 = concatenate([upconv1, enet_block1_2], axis=-1)
    conv2 = bottleneck(concat1, nb_filter=64, output_size=4, dilation_rate=(1, 1))

    upconv2 = UpSampling2D(size=(2, 2))(conv2)
    concat2 = concatenate([upconv2, conv1], axis=-1)
    conv3 = final_block(concat2, nb_filter=num_classes)

    # Создание модели
    model = Model(inputs=input_layer, outputs=conv3)
    return model

def initial_block(input_layer, nb_filter):
    # Layers for the initial block of the model
    convolution = Conv2D(nb_filter, (3, 3), padding='same')(input_layer)
    out = Conv2D(nb_filter, (3, 3), padding='same')(convolution)
    max_pool = MaxPooling2D(pool_size=(2, 2))(out)
    return max_pool

def bottleneck(input_layer, nb_filter, output_size, dilation_rate):
    # Layers for each bottleneck block
    convolution = Conv2D(nb_filter, (1, 1), dilation_rate=dilation_rate)(input_layer)
    out = Conv2D(nb_filter, (3, 3), padding='same', dilation_rate=dilation_rate, activation='relu')(convolution)
    out = Conv2D(nb_filter * output_size, (1, 1))(out)
    out = MaxPooling2D(pool_size=(2, 2))(out)
    return out

def final_block(input_layer, nb_filter):
    # Layers for the final block of the model
    convolution = Conv2D(nb_filter, (3, 3), padding='same')(input_layer)
    out = Conv2D(nb_filter, (3, 3), padding='same')(convolution)
    return out

def train_enet(train_paths, test_paths, image_dir, mask_dir, input_shape, num_classes, batch_size, num_epochs):
    # Создание модели
    model = create_enet_model(input_shape, num_classes)

    # Компиляция модели
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    # Создание списков путей для тренировочных и тестовых данных
    train_image_paths = glob.glob(os.path.join(image_dir, train_paths))
    train_mask_paths = glob.glob(os.path.join(mask_dir, train_paths))
    test_image_paths = glob.glob(os.path.join(image_dir, test_paths))
    test_mask_paths = glob.glob(os.path.join(mask_dir, test_paths))

    # Получение количества тренировочных и тестовых изображений
    num_train_samples = len(train_image_paths)
    num_test_samples = len(test_image_paths)

    # Цикл обучения модели
    for epoch in range(num_epochs):
        # Получение случайного порядка для тренировочных изображений
        random_order = random.sample(range(num_train_samples), num_train_samples)

        # Обучение модели на тренировочных изображениях в указанном порядке
        for i in range(0, num_train_samples, batch_size):
            batch_indices = random_order[i:min(i + batch_size, num_train_samples)]
            batch_images = []
            batch_masks = []
            for index in batch_indices:
                image_path = train_image_paths[index]
                mask_path = train_mask_paths[index]
                image, mask = load_data(image_path, mask_path)
                batch_images.append(image)
                batch_masks.append(mask)
            batch_images = np.array(batch_images)
            batch_masks = np.array(batch_masks)

            model.train_on_batch(batch_images, batch_masks)

        # Оценка модели на тестовых изображениях после каждой эпохи
        test_images = []
        test_masks = []
        for index in range(num_test_samples):
            image_path = test_image_paths[index]
            mask_path = test_mask_paths[index]
            image, mask = load_data(image_path, mask_path)
            test_images.append(image)
            test_masks.append(mask)
        test_images = np.array(test_images)
        test_masks = np.array(test_masks)

        loss = model.evaluate(test_images, test_masks, verbose=0)
        print('Epoch: {}, Validation loss: {:.4f}'.format(epoch+1, loss))