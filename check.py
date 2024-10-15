#!/usr/bin/env python3

from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

IMG_FILE = 'D00_dataset/training/scissors/scissors_0003.jpg'
PLT_ROW = 1
PLT_COL = 4


def plot(title, img, datagrn):
    plt.figure(title)
    i = 0
    for data in datagen.flow(img, batch_size=1):
        plt.subplot(PLT_ROW, PLT_COL, i + 1)
        plt.axis('off')
        plt.imshow(array_to_img(data[0]))
        i += 1
        if i == PLT_ROW * PLT_COL:
            break
        plt.show()
#p86

img = load_img(IMG_FILE, target_size=(160, 160))

img = img_to_array(img)

img = img.reshape((1,) + img.shape)


datagen = ImageDataGenerator(rotation_range=30)
plot('rotation', img, datagen)


datagen = ImageDataGenerator(width_shift_range=0.2)
plot('width_shift', img, datagen)

datagen = ImageDataGenerator(height_shift_range=0.2)
plot('height_shift', img, datagen)

datagen = ImageDataGenerator(shear_range=30)
plot('shear', img, datagen)

datagen = ImageDataGenerator(zoom_range=[0.7, 1.3])
plot('zoom', img, datagen)

datagen = ImageDataGenerator(horizontal_flip=True)
plot('horizontal_flip', img, datagen)

datagen = ImageDataGenerator(vertical_flip=True)
plot('vertical_flip', img, datagen)

datagen = ImageDataGenerator(
    rotation_range = 30,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 30,
    zoom_range = [0.7, 1.3],
    horizontal_flip = True,
    vertical_flip = True
)
plot('all', img, datagen)
