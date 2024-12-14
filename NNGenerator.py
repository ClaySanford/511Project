#This file belongs in the base ApproxTrain folder. This file generates all other required files. This file will not work nicely with other multipliers.
import os
import sys

#The following multi-line comment was an attempt to make the individual NN customizable; however, I was unable to get it to work, and instead left the NN as 2 2D and then 2 dense layers. If I had more time, this is probably the first thing I would elaborate.
"""C2DCount = input("Enter number of 2D Convolutional layers (default 2): ")
try:
    C2DCount = int(C2DCount)
except ValueError:
    C2DCount = 2

DenseCount = 4 - C2DCount """

finalScript = open("./injected_error.py", 'w')

finalScript.write("""
import tensorflow as tf
import tensorflow_datasets as tfds
import sys
from python.keras.layers.am_convolutional import AMConv2D
from python.keras.layers.amdenselayer import denseam
tf.random.set_seed(0)

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)
def normalize_img(image, label):
  #Normalizes images: `uint8` -> `float32`.
  return tf.cast(image, tf.float32) / 255., label

lut_file = "lut/ERRORINJ_10.bin"
ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

model0 = tf.keras.models.Sequential([
 tf.keras.layers.Input(shape=(28, 28, 1)),
 #AMConv2D(filters=32, kernel_size=5, padding='same', activation='relu', mant_mul_lut=lut_file),
 #tf.keras.layers.Conv2D()
 tf.keras.layers.Conv2D(filters=32, kernel_size=5, padding='same', activation='relu'),
 tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2),padding='same'),
 #AMConv2D(filters=32, kernel_size=5, padding='same', activation='relu', mant_mul_lut=lut_file),
 tf.keras.layers.Conv2D(filters=32, kernel_size=5, padding='same', activation='relu'),
 tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
 tf.keras.layers.Flatten(),
 tf.keras.layers.Dense(1024, activation='relu'),
 #denseam(1024, activation='relu', mant_mul_lut=lut_file),
 tf.keras.layers.Dropout(0.4),
 tf.keras.layers.Dense(10, activation='softmax'),
 #denseam(10, activation='softmax', mant_mul_lut=lut_file)
])
model0.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

print('No approximation:')

model0.fit(
    ds_train,
    epochs=2,
    validation_data=ds_test,
)

print(' ')
print('First layer approximation:')
print(' ')

model1 = tf.keras.models.Sequential([
 tf.keras.layers.Input(shape=(28, 28, 1)),
 AMConv2D(filters=32, kernel_size=5, padding='same', activation='relu', mant_mul_lut=lut_file),
 #tf.keras.layers.Conv2D()
 #tf.keras.layers.Conv2D(filters=32, kernel_size=5, padding='same', activation='relu'),
 tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2),padding='same'),
 #AMConv2D(filters=32, kernel_size=5, padding='same', activation='relu', mant_mul_lut=lut_file),
 tf.keras.layers.Conv2D(filters=32, kernel_size=5, padding='same', activation='relu'),
 tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
 tf.keras.layers.Flatten(),
 tf.keras.layers.Dense(1024, activation='relu'),
 #denseam(1024, activation='relu', mant_mul_lut=lut_file),
 tf.keras.layers.Dropout(0.4),
 tf.keras.layers.Dense(10, activation='softmax'),
 #denseam(10, activation='softmax', mant_mul_lut=lut_file)
])
model1.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model1.fit(
    ds_train,
    epochs=2,
    validation_data=ds_test,
)

print(' ')
print('Second layer approximation:')
print(' ')

model2 = tf.keras.models.Sequential([
 tf.keras.layers.Input(shape=(28, 28, 1)),
 #AMConv2D(filters=32, kernel_size=5, padding='same', activation='relu', mant_mul_lut=lut_file),
 #tf.keras.layers.Conv2D()
 tf.keras.layers.Conv2D(filters=32, kernel_size=5, padding='same', activation='relu'),
 tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2),padding='same'),
 AMConv2D(filters=32, kernel_size=5, padding='same', activation='relu', mant_mul_lut=lut_file),
 #tf.keras.layers.Conv2D(filters=32, kernel_size=5, padding='same', activation='relu'),
 tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
 tf.keras.layers.Flatten(),
 tf.keras.layers.Dense(1024, activation='relu'),
 #denseam(1024, activation='relu', mant_mul_lut=lut_file),
 tf.keras.layers.Dropout(0.4),
 tf.keras.layers.Dense(10, activation='softmax'),
 #denseam(10, activation='softmax', mant_mul_lut=lut_file)
])
model2.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model2.fit(
    ds_train,
    epochs=2,
    validation_data=ds_test,
)

print(' ')
print('Third layer approximation:')
print(' ')

model3 = tf.keras.models.Sequential([
 tf.keras.layers.Input(shape=(28, 28, 1)),
 #AMConv2D(filters=32, kernel_size=5, padding='same', activation='relu', mant_mul_lut=lut_file),
 #tf.keras.layers.Conv2D()
 tf.keras.layers.Conv2D(filters=32, kernel_size=5, padding='same', activation='relu'),
 tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2),padding='same'),
 #AMConv2D(filters=32, kernel_size=5, padding='same', activation='relu', mant_mul_lut=lut_file),
 tf.keras.layers.Conv2D(filters=32, kernel_size=5, padding='same', activation='relu'),
 tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
 tf.keras.layers.Flatten(),
 #tf.keras.layers.Dense(1024, activation='relu'),
 denseam(1024, activation='relu', mant_mul_lut=lut_file),
 tf.keras.layers.Dropout(0.4),
 tf.keras.layers.Dense(10, activation='softmax'),
 #denseam(10, activation='softmax', mant_mul_lut=lut_file)
])
model3.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model3.fit(
    ds_train,
    epochs=2,
    validation_data=ds_test,
)

print(' ')
print('Fourth layer approximation:')
print(' ')

model4 = tf.keras.models.Sequential([
 tf.keras.layers.Input(shape=(28, 28, 1)),
 #AMConv2D(filters=32, kernel_size=5, padding='same', activation='relu', mant_mul_lut=lut_file),
 #tf.keras.layers.Conv2D()
 tf.keras.layers.Conv2D(filters=32, kernel_size=5, padding='same', activation='relu'),
 tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2),padding='same'),
 #AMConv2D(filters=32, kernel_size=5, padding='same', activation='relu', mant_mul_lut=lut_file),
 tf.keras.layers.Conv2D(filters=32, kernel_size=5, padding='same', activation='relu'),
 tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
 tf.keras.layers.Flatten(),
 tf.keras.layers.Dense(1024, activation='relu'),
 #denseam(1024, activation='relu', mant_mul_lut=lut_file),
 tf.keras.layers.Dropout(0.4),
 #tf.keras.layers.Dense(10, activation='softmax'),
 denseam(10, activation='softmax', mant_mul_lut=lut_file)
])
model4.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model4.fit(
    ds_train,
    epochs=2,
    validation_data=ds_test,
)

print(' ')
print('Full approximation:')
print(' ')

modelFull = tf.keras.models.Sequential([
 tf.keras.layers.Input(shape=(28, 28, 1)),
 AMConv2D(filters=32, kernel_size=5, padding='same', activation='relu', mant_mul_lut=lut_file),
 #tf.keras.layers.Conv2D()
 #tf.keras.layers.Conv2D(filters=32, kernel_size=5, padding='same', activation='relu'),
 tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2),padding='same'),
 AMConv2D(filters=32, kernel_size=5, padding='same', activation='relu', mant_mul_lut=lut_file),
 #tf.keras.layers.Conv2D(filters=32, kernel_size=5, padding='same', activation='relu'),
 tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
 tf.keras.layers.Flatten(),
 #tf.keras.layers.Dense(1024, activation='relu'),
 denseam(1024, activation='relu', mant_mul_lut=lut_file),
 tf.keras.layers.Dropout(0.4),
 #tf.keras.layers.Dense(10, activation='softmax'),
 denseam(10, activation='softmax', mant_mul_lut=lut_file)
])
modelFull.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

modelFull.fit(
    ds_train,
    epochs=2,
    validation_data=ds_test,
)

""")

finalScript.flush()
os.system('python ./injected_error.py')