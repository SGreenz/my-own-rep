import os
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
import cv2

from skimage import measure
from skimage.io import imread, imsave, imshow
from skimage.transform import resize
from skimage.filters import gaussian
from skimage.morphology import dilation, disk
from skimage.draw import polygon, polygon_perimeter

CLASSES = 4

COLORS = ['black', 'blue', 'yellow','purple']

SAMPLE_SIZE = (256, 256)

OUTPUT_SIZE = (656, 1184)

def load_images(image, mask):
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image)
    image = tf.image.resize(image, OUTPUT_SIZE)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = image / 255.0

    mask = tf.io.read_file(mask)
    mask = tf.io.decode_png(mask)
   # mask = tf.image.rgb_to_grayscale(mask)
    mask = tf.image.resize(mask, OUTPUT_SIZE)
    mask = tf.image.convert_image_dtype(mask, tf.float32)

    masks = []

    for i in range(CLASSES):
        masks.append(tf.where(tf.equal(mask, float(i)), 1.0, 0.0))

    masks = tf.stack(masks, axis=2)
    masks = tf.reshape(masks, OUTPUT_SIZE + (CLASSES,))

    return image, masks

def augmentate_images(image, masks):
    random_crop = tf.random.uniform((), 0.3, 1)
    image = tf.image.central_crop(image, random_crop)
    masks = tf.image.central_crop(masks, random_crop)

    random_flip = tf.random.uniform((), 0, 1)
    if random_flip >= 0.5:
        image = tf.image.flip_left_right(image)
        masks = tf.image.flip_left_right(masks)

    image = tf.image.resize(image, SAMPLE_SIZE)
    masks = tf.image.resize(masks, SAMPLE_SIZE)

    return image, masks

images = sorted(glob.glob('/home/egorov_pa@RODINA.LOC/PROJECTS/NNthread/program/img1/*.jpg'))
masks = sorted(glob.glob('/home/egorov_pa@RODINA.LOC/PROJECTS/NNthread/program/masksCityscapes/*.png'))

images_dataset = tf.data.Dataset.from_tensor_slices(images)
masks_dataset = tf.data.Dataset.from_tensor_slices(masks)

dataset = tf.data.Dataset.zip((images_dataset, masks_dataset))

dataset = dataset.map(load_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)    #tf.data.experimental.AUTOTUNE
dataset = dataset.repeat(60)
dataset = dataset.map(augmentate_images, num_parallel_calls=tf.data.AUTOTUNE)

train_dataset = dataset.take(2800).cache()
test_dataset = dataset.skip(2800).take(100).cache()

train_dataset = train_dataset.batch(16)
test_dataset = test_dataset.batch(16)

def input_layer():
    return tf.keras.layers.Input(shape=SAMPLE_SIZE + (3,))

def downsample_block(filters, size, batch_norm=True):
    initializer = tf.keras.initializers.GlorotNormal()

    result = tf.keras.Sequential()

    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))

    if batch_norm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())
    return result

def upsample_block(filters, size, dropout=False):
    initializer = tf.keras.initializers.GlorotNormal()

    result = tf.keras.Sequential()

    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                        kernel_initializer=initializer, use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if dropout:
        result.add(tf.keras.layers.Dropout(0.25))

    result.add(tf.keras.layers.ReLU())
    return result

def output_layer(size):
    initializer = tf.keras.initializers.GlorotNormal()
    return tf.keras.layers.Conv2DTranspose(CLASSES, size, strides=2, padding='same',
                                           kernel_initializer=initializer, activation='sigmoid')

inp_layer = input_layer()

downsample_stack = [
    downsample_block(64, 4, batch_norm=False),
    downsample_block(128, 4),
    downsample_block(256, 4),
    downsample_block(512, 4),
    downsample_block(512, 4),
    downsample_block(512, 4),
    downsample_block(512, 4),
]

upsample_stack = [
    upsample_block(512, 4, dropout=True),
    upsample_block(512, 4, dropout=True),
    upsample_block(512, 4, dropout=True),
    upsample_block(256, 4),
    upsample_block(128, 4),
    upsample_block(64, 4)
]

out_layer = output_layer(4)

# Реализуем skip connections
x = inp_layer

downsample_skips = []

for block in downsample_stack:
    x = block(x)
    downsample_skips.append(x)

downsample_skips = reversed(downsample_skips[:-1])

for up_block, down_block in zip(upsample_stack, downsample_skips):
    x = up_block(x)
    x = tf.keras.layers.Concatenate()([x, down_block])

out_layer = out_layer(x)

unet_like = tf.keras.Model(inputs=inp_layer, outputs=out_layer)

# tf.keras.utils.plot_model(unet_like, show_shapes=True, dpi=72)

def dice_mc_metric(a, b):
    a = tf.unstack(a, axis=3)
    b = tf.unstack(b, axis=3)

    dice_summ = 0

    for i, (aa, bb) in enumerate(zip(a, b)):
        numenator = 2 * tf.math.reduce_sum(aa * bb) + 1
        denomerator = tf.math.reduce_sum(aa + bb) + 1
        dice_summ += numenator / denomerator

    avg_dice = dice_summ / CLASSES

    return avg_dice


def dice_mc_loss(a, b):
    return 1 - dice_mc_metric(a, b)


def dice_bce_mc_loss(a, b):
    return 0.3 * dice_mc_loss(a, b) + tf.keras.losses.binary_crossentropy(a, b)

unet_like.compile(optimizer='adam', loss=[dice_bce_mc_loss], metrics=[dice_mc_metric])

history_dice = unet_like.fit(train_dataset, validation_data=test_dataset, epochs=30, initial_epoch=0)

unet_like.save_weights('/home/egorov_pa@RODINA.LOC/PROJECTS/NNthread/program/networks/unet_like')

#unet_like.load_weights('/home/egorov_pa@RODINA.LOC/PROJECTS/NNthread/program/networks/unet_like')

#rgb_colors = [
#    (0, 0, 0),
#    (255, 0, 0),
#    (0, 255, 0),
#    (255, 255, 0)
#]

#frames = sorted(glob.glob('/home/egorov_pa@RODINA.LOC/PROJECTS/NNthread/program/images/*.jpg'))

#for filename in frames:
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