#This file belongs in the base ApproxTrain folder. This file generates all other required files. This file will not work nicely with other multipliers.
import os
import sys

#Write the error inl file
SelVariance = input("Enter variance (leave blank for default 15):")
try:
    float(SelVariance)
except ValueError:
    SelVariance = "15.0"
inlFile = open("./lut/ErrorInject.inl", 'w')
inlFile.write("""

//
//Since I don't actually care about real multiplication, I need to define an INL that just adds some error to the normal multiplication. The error should be customizable. 
//My idea is to use the gaussian distribution; this allows for me to use the actually calculated product as the mean, and then a customizable variance.
//This WILL NOT speed up a DNN; this is not designed to speed up a DNN; this is just designed to introduce error.
#include <random>
//



//
float ErrorInject(float Af, float Bf, float variance=""" + SelVariance + """)
{
	float sum = Af * Bf;
	std::default_random_engine generator;
	std::normal_distribution<float> distribution(sum, variance);
	return distribution(generator);

}
//

""")
inlFile.flush()
inlFile.close()


#Write the lut_gen.cc file
ccFile = open("./lut/lut_gen.cc", 'w')
ccFile.write("""
//This file belongs in the lut folder. This is the file that generates the lookup table when the script is called.

#include <cstdio>
#include <cstdint>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <bitset>
#include <string>
#include <cmath>
void floatToBinary(float f, std::string& str)
{

    union { float f; uint32_t i; } u;
    u.f = f;
    str.clear();

    for (int i = 0; i < 32; i++)
    {
        if (u.i % 2)  str.push_back('1');
        else str.push_back('0');
        u.i >>= 1;
    }

    // Reverse the string since now it's backwards
    std::string temp(str.rbegin(), str.rend());
    str = temp;
}
#ifdef FMBM16_MULTIPLIER
#define MULTIPLY(a,b) FPmultMBM_fast16((a),(b));
#include "FPmultMBM_fast16.inl"
#define MANTISSA_BITWIDTH 7
std::string lut_save = "MBM_7.bin";
#elif ERRORINJECTOR
#define MULTIPLY(a,b) ErrorInject((a),(b));
#include "ErrorInject.inl"
#define MANTISSA_BITWIDTH 10
std::string lut_save = "ERRORINJ_10.bin";
#elif FMBM14_MULTIPLIER
#define MULTIPLY(a,b) FPmultMBM_fast14((a),(b));
#include "FPmultMBM_fast14.inl"
#define MANTISSA_BITWIDTH 5
std::string lut_save = "MBM_5.bin";
#elif FMBM12_MULTIPLIER
#define MULTIPLY(a,b) FPmultMBM_fast12((a),(b));
#include "FPmultMBM_fast12.inl"
#define MANTISSA_BITWIDTH 3
std::string lut_save = "MBM_3.bin";
#elif FMBM10_MULTIPLIER
#define MULTIPLY(a,b) FPmultMBM_fast10((a),(b));
#include "FPmultMBM_fast10.inl"
#define MANTISSA_BITWIDTH 1
std::string lut_save = "MBM_1.bin";
#elif MITCHEL16_MULTIPLIER
#define MULTIPLY(a,b) FPmultMitch_fast16((a),(b));
#include "Mitchell_16.inl"
#define MANTISSA_BITWIDTH 7
std::string lut_save = "MIT_7.bin";
#elif MITCHEL14_MULTIPLIER
#define MULTIPLY(a,b) FPmultMitch_fast14((a),(b));
#include "Mitchell_14.inl"
#define MANTISSA_BITWIDTH 5
std::string lut_save = "MIT_5.bin";
#elif MITCHEL12_MULTIPLIER
#define MULTIPLY(a,b) FPmultMitch_fast12((a),(b));
#include "Mitchell_12.inl"
#define MANTISSA_BITWIDTH 3
std::string lut_save = "MIT_3.bin";
#elif MITCHEL10_MULTIPLIER
#define MULTIPLY(a,b) FPmultMitch_fast10((a),(b));
#include "Mitchell_10.inl"
#define MANTISSA_BITWIDTH 1
std::string lut_save = "MIT_1.bin";
#elif BFLOAT
#define MULTIPLY(a,b) bfloat16mul((a),(b));
#include "bfloat.inl"
#define MANTISSA_BITWIDTH 7
std::string lut_save = "ACC_7.bin";
#elif ZEROS
#define MULTIPLY(a,b) 0;
#define MANTISSA_BITWIDTH 7
std::string lut_save = "ZEROS_7.bin";
#endif

#define EMPTYFP32 0x00000000
//#define SIGN_MASK_ 0x80000000
#define EXPONENT127 0x3f800000
#define EXPONENT_MASK_ 0x7f800000
#define MANTISSA_MASK_ ((uint32_t(pow(2,MANTISSA_BITWIDTH))-1) << (23-MANTISSA_BITWIDTH))
// implementation for approximate mantissa multiplications lookup table generation
int main() {
    // create a and b
    float a = 0;
    float b = 0;
    // cast to uint32_t
    uint32_t  at = *(uint32_t*)&a;
    uint32_t  bt = *(uint32_t*)&b;
    // FP32 with bits set to all zeros
    at = at & EMPTYFP32;
    bt = bt & EMPTYFP32;
    // set sign to 0 or 1
    // set exponents A B C (output of A*B) should be normal case
    // 0b0011 1111 1000 0000 0000 0000 0000 0000 Biased exponent = 127
    at = at | EXPONENT127;
    bt = bt | EXPONENT127;



    char* lut_save_name = &lut_save[0];
    FILE* f = fopen(lut_save_name, "wb");
    for (uint32_t i = 0; i < uint32_t(pow(2, MANTISSA_BITWIDTH)); ++i) {
        for (uint32_t j = 0; j < uint32_t(pow(2, MANTISSA_BITWIDTH)); ++j) {
            uint32_t newat = at | (i << (23 - MANTISSA_BITWIDTH));
            uint32_t newbt = bt | (j << (23 - MANTISSA_BITWIDTH));
            float newa = *(float*)&newat;
            float newb = *(float*)&newbt;
            float c = MULTIPLY(newa, newb);
            uint32_t ct = *(uint32_t*)&c;
            uint8_t MANTISSA = (ct & MANTISSA_MASK_) >> (23 - MANTISSA_BITWIDTH);
            uint32_t c_exp = ct & EXPONENT_MASK_;
            uint32_t un_normalized_exp = ((EXPONENT127 >> 23) + (EXPONENT127 >> 23) - 127) << 23;
            uint8_t carry = 0;
            if (un_normalized_exp < c_exp)
                carry = 0x80;
            uint8_t result = carry | MANTISSA;
            fwrite(&result, sizeof(uint8_t), 1, f);
        }
    }

    fclose(f);
    return 0;
}
""")

ccFile.flush()
ccFile.close()

shFile = open("./lut/lut_gen.sh", 'w')
shFile.write("""
#This file belongs in the lut folder. This is the script that calls for the lookup tables to be generated.

MULTIPLIER=(
    "FMBM16_MULTIPLIER"
    "FMBM14_MULTIPLIER"
    "FMBM12_MULTIPLIER"
    "FMBM10_MULTIPLIER"
    "MITCHEL16_MULTIPLIER"
    "MITCHEL14_MULTIPLIER"
    "MITCHEL12_MULTIPLIER"
    "MITCHEL10_MULTIPLIER"
    "BFLOAT"
    "ZEROS"
    "ERRORINJECTOR"
    )
    for i in "${MULTIPLIER[@]}"; do
        g++ -D$i ./lut/lut_gen.cc
        ./a.out
        done""")

shFile.flush()
shFile.close()

os.system('bash ./lut/lut_gen.sh')

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