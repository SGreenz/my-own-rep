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
from skimage import morphology

CLASSES = 4
# CLASSES = 4

SAMPLE_SIZE = (256, 256)

OUTPUT_SIZE = (614, 1096)

def vide_tuning(focus):
    os.system(f'v4l2-ctl -c focus_auto=0')
    time.sleep(0.1)
    os.system(f'v4l2-ctl -c focus_absolute=1')
    time.sleep(0.1)
    os.system(f'v4l2-ctl -c focus_absolute={focus}')
    time.sleep(0.1)

def crop(img,tl,br):
    return img[tl[1]:br[1], tl[0]:br[0]]

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

# def draw_text(img,x,y,color,str):
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     org = (x, y)
#     fontScale = 0.5
#     thickness = 1
#     img = cv2.putText(img, str, org, font, fontScale, color, thickness, cv2.LINE_AA)

def key_pressed_handler(focus,frame,cap):
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # esc key ends process
        cap.release()
        return True

    if k == 56:  # esc key ends process
        focus += 4
        os.system(f'v4l2-ctl -c focus_absolute={focus}')
        print(f"focus {focus}")

    if k == 50:  # esc key ends process13248
        focus -= 4
        os.system(f'v4l2-ctl -c focus_absolute={focus}')
        print(f"focus {focus}")

    if k == 32:  # esc key ends process8
        tst = time.monotonic();
        cv2.imwrite('cap/' + f"{tst}.jpg", frame)

    if k != 255:
        print(f"key code={k}")
    return False

unet_like.compile(optimizer='adam', loss=[dice_bce_mc_loss], metrics=[dice_mc_metric])
unet_like.load_weights('/home/egorov_pa@RODINA.LOC/PROJECTS/NNthread/program/networks/unet_like').expect_partial()

# frames = sorted(glob.glob('/home/egorov_pa@RODINA.LOC/PROJECTS/NNthread/images/*.jpg'))
def realtime_test(focus):
    rgb_colors = [
        (0, 0, 0),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255)
    ]

    window_name = 'Camera'
    window_name_2 = 'addw'

    cv2.namedWindow(window_name)  # Create a named window
    cv2.moveWindow(window_name, 800, 400)  # Move it to (40,30)
    cv2.namedWindow(window_name_2)  # Create a named window
    cv2.moveWindow(window_name_2, 800, 400)  # Move it to (40,30)

    cap = cv2.VideoCapture("/dev/video0")
    while cap.isOpened():
        # frame = imread(filename)
        ret, frame = cap.read()
        
        sample = resize(frame, SAMPLE_SIZE)
        # print(sample.reshape((1,) + SAMPLE_SIZE + (3,)))       
        predict = unet_like.predict(sample.reshape((1,) + SAMPLE_SIZE + (3,)))         
        predict = predict.reshape(SAMPLE_SIZE + (CLASSES,))

        cv2.imshow(window_name_2, predict)        

        scale = frame.shape[0] / SAMPLE_SIZE[0], frame.shape[1] / SAMPLE_SIZE[1]
        frame = (frame / 1.5).astype(np.uint8)
        
        for channel in range(1, CLASSES):
            contour_overlay = np.zeros((frame.shape[0], frame.shape[1]))
            contours = measure.find_contours(np.array(predict[:, :, channel]))

            try:
                for contour in contours:
                    rr, cc = polygon_perimeter(contour[:, 0] * scale[0],
                                           contour[:, 1] * scale[1],
                                           shape=contour_overlay.shape)


                    contour_overlay[rr, cc] = 1

                contour_overlay = dilation(contour_overlay, disk(1))
                frame[contour_overlay == 1] = rgb_colors[channel]
                
                # if channel == 1:
                #     print("синий")
                # if channel == 2:
                #     print("зелёный")
                # if channel == 3:
                #     print("красный")

            except:
                pass
        cv2.imshow(window_name, frame)
        if key_pressed_handler(focus, frame, cap):
            break

    if not cap.isOpened():
        print("Нет камеры")
    cv2.destroyAllWindows()
    # imsave(f'/home/egorov_pa@RODINA.LOC/PROJECTS/NNthread/res/{os.path.basename(filename)}', frame)
#def main():
 #   focus = 90
 #   realtime_test(focus)
#main   
focus = 90
realtime_test(focus) 