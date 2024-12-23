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

print("\nNo approximation:\n")

model0.fit(
    ds_train,
    epochs=2,
    validation_data=ds_test,
)

print("\nFirst layer approximation:\n")

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

print("\nSecond layer approximation:\n")
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

print("\nThird layer approximation:\n")

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

print("\nFourth layer approximation:\n")
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

print("\nFull approximation:\n")
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
