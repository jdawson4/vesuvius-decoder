# Author: Jacob Dawson
#
# Note: I'm looking at
# www.kaggle.com/code/fchollet/keras-starter-kit-unet-train-on-full-dataset
# to begin with. François Chollet (look at that, the keyboard accepted my
# accent mark!) is an industry giant, and the mind behind keras, my preferred
# DL library
#
# FChollet has written a ton of preprocessing code for this project which I
# was struggling with, and I wanted to start with it because it seems that he's
# inviting people to use his code as a jumping-off point for writing their own
# keras models. What I really want to experiment with/get to is the ability to
# try different network structures, so I'll largely be adapting from the
# notebook that monsieur Chollet left on kaggle

import numpy as np
import tensorflow as tf
from tensorflow import keras
import PIL
import tqdm

import glob
import time
import gc

from architecture import *

# Data config
DATA_DIR = "../vesuvius-data/"
BUFFER = 32  # Half-size of papyrus patches we'll use as model inputs
Z_DIM = 64  # Number of slices in the z direction. Max value is 64 - Z_START
Z_START = 0  # Offset of slices in the z direction
SHARED_HEIGHT = 2400  # Height to resize all papyrii, originally 4000 but my computer is much worse than FChollet's so I might have to downsize

# Model config
BATCH_SIZE = 32
USE_MIXED_PRECISION = False
USE_JIT_COMPILE = False

if USE_MIXED_PRECISION:
    keras.mixed_precision.set_global_policy("mixed_float16")

physical_devices = tf.config.experimental.list_physical_devices('GPU')
num_gpus = len(physical_devices)
print(f"Num GPUs: {num_gpus}")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

seed = 3
tf.random.set_seed(seed)

epochs = 50 # originally 20
steps_per_epoch = 1000 * BATCH_SIZE # originally 1000
learnRate = 0.001 # default: 0.001
momentum = 0.9 # default: 0.9
epoch_interval = 5


def resize(img):
    current_width, current_height = img.size
    aspect_ratio = current_width / current_height
    new_width = int(SHARED_HEIGHT * aspect_ratio)
    new_size = (new_width, SHARED_HEIGHT)
    img = img.resize(new_size)
    return img


def load_mask(split, index):
    img = PIL.Image.open(f"{DATA_DIR}/{split}/{index}/mask.png").convert("1")
    img = resize(img)
    return tf.convert_to_tensor(img, dtype="bool")


def load_labels(split, index):
    img = PIL.Image.open(f"{DATA_DIR}/{split}/{index}/inklabels.png")
    img = resize(img)
    return tf.convert_to_tensor(img, dtype="bool")


mask = load_mask(split="train", index=1)
labels = load_labels(split="train", index=1)

mask_test_a = load_mask(split="test", index="a")
mask_test_b = load_mask(split="test", index="b")

mask_train_1 = load_mask(split="train", index=1)
labels_train_1 = load_labels(split="train", index=1)

mask_train_2 = load_mask(split="train", index=2)
labels_train_2 = load_labels(split="train", index=2)

mask_train_3 = load_mask(split="train", index=3)
labels_train_3 = load_labels(split="train", index=3)

"""print(f"mask_test_a: {mask_test_a.shape}")
print(f"mask_test_b: {mask_test_b.shape}")
print("-")
print(f"mask_train_1: {mask_train_1.shape}")
print(f"labels_train_1: {labels_train_1.shape}")
print("-")
print(f"mask_train_2: {mask_train_2.shape}")
print(f"labels_train_2: {labels_train_2.shape}")
print("-")
print(f"mask_train_3: {mask_train_3.shape}")
print(f"labels_train_3: {labels_train_3.shape}")"""


def load_volume(split, index):
    # Load the 3d x-ray scan, one slice at a time
    z_slices_fnames = sorted(
        glob.glob(f"{DATA_DIR}/{split}/{index}/surface_volume/*.tif")
    )[Z_START : Z_START + Z_DIM]
    z_slices = []
    for filename in z_slices_fnames:
        img = PIL.Image.open(filename)
        img = resize(img)
        z_slice = np.array(img, dtype="uint16")
        z_slices.append(z_slice)
    return tf.cast(tf.stack(z_slices, axis=-1), dtype="uint16")

gc.collect()


volume = load_volume(split="train", index=1)
# print(f"volume_train_1: {volume_train_1.shape}, {volume_train_1.dtype}")

volume = tf.concat([volume, load_volume(split="train", index=2)], axis=1)
# print(f"volume_train_2: {volume_train_2.shape}, {volume_train_2.dtype}")

volume = tf.concat([volume, load_volume(split="train", index=3)], axis=1)
# print(f"volume_train_3: {volume_train_3.shape}, {volume_train_3.dtype}")

#volume = tf.concat([volume_train_1, volume_train_2, volume_train_3], axis=1)
# so what has happened now is that we have a 2D image containing our ENTIRE
# train set, shaped [4000,8417,20] (that is, a 2D image 20 layers deep!), which
# contains pages 1, 2, and 3. Similarly, our labels and mask (set below) will
# contain all three pages as well!
# note that the size listed above depends on SHARED_HEIGHT, which I might mess
# with because my computer is getting OOM errors
print(f"total volume: {volume.shape}, {volume.dtype}")
print(f"Volume's max: {tf.math.reduce_max(volume)}, min: {tf.math.reduce_min(volume)},mean: {tf.math.reduce_mean(volume)}\n")
#print(f"total volume: {volume.shape}")

#del volume_train_1
#del volume_train_2
#del volume_train_3

gc.collect()

labels = tf.concat([labels_train_1, labels_train_2, labels_train_3], axis=1)
print(f"labels: {labels.shape}, {labels.dtype}")

mask = tf.concat([mask_train_1, mask_train_2, mask_train_3], axis=1)
print(f"mask: {mask.shape}, {mask.dtype}")

# Free up memory
del labels_train_1
del labels_train_2
del labels_train_3
del mask_train_1
del mask_train_2
del mask_train_3

# val_location = (1300, 1000)
# val_zone_size = (600, 2000)

val_location = (
    int((1300 / 4000) * SHARED_HEIGHT),
    int((1300 / 8417) * volume.shape[1]),
)
val_zone_size = (
    int((600 / 4000) * SHARED_HEIGHT),
    int((2000 / 8417) * volume.shape[1]),
)
print(val_location, val_zone_size)


def sample_random_location(shape):
    random_train_x = tf.random.uniform(
        shape=(), minval=BUFFER, maxval=shape[0] - BUFFER - 1, dtype="int32"
    )
    random_train_y = tf.random.uniform(
        shape=(), minval=BUFFER, maxval=shape[1] - BUFFER - 1, dtype="int32"
    )
    random_train_location = tf.stack([random_train_x, random_train_y])
    return random_train_location


def is_in_masked_zone(location, mask):
    return mask[location[0], location[1]]


sample_random_location_train = lambda x: sample_random_location(mask.shape)
is_in_mask_train = lambda x: is_in_masked_zone(x, mask)

def is_in_val_zone(location, val_location, val_zone_size):
    #print(location)
    x = location[0]
    y = location[1]
    # seems like there's a problem here with using these tensor slices as
    # ints. Ensure that we're using just the integer values!
    x_match = tf.math.logical_and(
        tf.math.less_equal(tf.constant(val_location[0] - BUFFER), x),
        tf.math.less_equal(x, tf.constant(val_location[0] + val_zone_size[0] + BUFFER))
    )
    y_match = tf.math.logical_and(
        tf.math.less_equal(tf.constant(val_location[1] - BUFFER), y),
        tf.math.less_equal(y, tf.constant(val_location[1] + val_zone_size[1] + BUFFER))
    )
    return tf.get_static_value(tf.logical_and(x_match, y_match))


def is_proper_train_location(location):
    #print(location)
    return not (is_in_val_zone(location, val_location=val_location, val_zone_size=val_zone_size) and is_in_mask_train(location))


train_locations_ds = tf.data.Dataset.from_tensor_slices([0]).repeat(steps_per_epoch).map(sample_random_location_train, num_parallel_calls=tf.data.AUTOTUNE)
#for location in train_locations_ds.take(1):
#    print(location)
#print(f"CARDINALITY OF TRAIN LOCATIONS DATASET BEFORE FILTER: {train_locations_ds.cardinality().numpy()}")
train_locations_ds = train_locations_ds.filter(is_proper_train_location)
#print(f"CARDINALITY OF TRAIN LOCATIONS DATASET AFTER FILTER: {train_locations_ds.cardinality().numpy()}")

#for location in train_locations_ds.take(1):
#    print("NANs in location?", tf.math.reduce_any(tf.math.is_nan(location)))
#    print(location)

#print("HERE1")

def extract_subvolume(location, volume):
    x = location[0]
    y = location[1]
    subvolume = volume[x - BUFFER : x + BUFFER, y - BUFFER : y + BUFFER, :]
    #subvolume = tf.cast(subvolume, dtype="float32") / 65535.0
    subvolume = tf.cast(subvolume, dtype="float32")
    return subvolume


def extract_labels(location, labels):
    x = location[0]
    y = location[1]
    label = labels[x - BUFFER : x + BUFFER, y - BUFFER : y + BUFFER]
    label = tf.cast(label, dtype="float32")
    label = tf.expand_dims(label, axis=-1)
    return label


def extract_subvolume_and_label(location):
    subvolume = extract_subvolume(location, volume)
    label = extract_labels(location, labels)
    return subvolume, label

#print("HERE2")


shuffle_buffer_size = BATCH_SIZE * 4

train_ds = train_locations_ds.map(
    extract_subvolume_and_label, num_parallel_calls=tf.data.AUTOTUNE
)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE).batch(BATCH_SIZE)

#print("HERE3")

for subvolume_batch, label_batch in train_ds.take(1):
    print(f"subvolume shape: {subvolume_batch.shape[1:]}")
    print(f"label_batch shape: {label_batch.shape[1:]}")
    #print("NANs in subvolume_batch?", tf.math.reduce_any(tf.math.is_nan(subvolume_batch)))

t0 = time.time()
n = steps_per_epoch//5
#show_value = True
for subvolume, label in train_ds.take(n):
    #if show_value:
    #    print(subvolume, label)
    #show_value = False
    #print("NANs in train subvolume?", tf.math.reduce_any(tf.math.is_nan(subvolume)))
    pass
print(f"Time per batch: {(time.time() - t0) / n:.4f}s")

#print("HERE4")

val_locations_stride = BUFFER
val_locations = []
for x in range(
    val_location[0], val_location[0] + val_zone_size[0], val_locations_stride
):
    for y in range(
        val_location[1], val_location[1] + val_zone_size[1], val_locations_stride
    ):
        val_locations.append((x, y))

val_locations_ds = tf.data.Dataset.from_tensor_slices(val_locations).filter(
    is_in_mask_train
)
val_ds = val_locations_ds.map(
    extract_subvolume_and_label, num_parallel_calls=tf.data.AUTOTUNE
)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE).batch(BATCH_SIZE)

#print("HERE5")


def trivial_baseline(dataset):
    total = 0
    matches = 0.0
    for _, batch_label in dataset:
        #print(batch_label)
        #print("NANS HERE?", tf.reduce_sum(batch_label), tf.reduce_prod(tf.shape(batch_label)))
        matches += tf.cast(tf.reduce_sum(batch_label), dtype = "float32")
        total += tf.cast(tf.reduce_prod(tf.shape(batch_label)), dtype = "float32")
    #print("NANS HERE?", matches, total)
    #print("OR HERE:", tf.cast(1.0 - (matches / total), dtype="float32"))
    return tf.cast(1.0 - (matches / total), dtype="float32")


score = trivial_baseline(val_ds).numpy()
print(f"Best validation score achievable trivially: {score * 100:.2f}% accuracy") # NANs here?

for subvolume, label in val_ds.take(1):
    print("VAL_DS LOOKS LIKE:", subvolume.shape, label.shape)

#print("HERE6")

# THIS IS CREATING NANs
augmenter = keras.Sequential(
    [
        #keras.Input((BUFFER * 2, BUFFER * 2, Z_DIM), dtype='float32'),
        keras.layers.RandomContrast(0.2),
    ]
)


def augment_train_data(data, label):
    data = tf.cast(augmenter(tf.cast(data,dtype="float32")), dtype="float32")
    return data, label


augmented_train_ds = train_ds.map(
    augment_train_data, num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

for subvolume, label in train_ds.take(1):
    print("NANs in train subvolume?", tf.math.reduce_any(tf.math.is_nan(subvolume)))
    #print("NANs in label?", tf.math.reduce_any(tf.math.is_nan(label)))
    #print("TRAIN_DS LOOKS LIKE:", subvolume.shape, label.shape)
for subvolume, label in augmented_train_ds.take(1):
    print("NANs in augmented subvolume?", tf.math.reduce_any(tf.math.is_nan(subvolume)))
    #print("NANs in label?", tf.math.reduce_any(tf.math.is_nan(label)))
    #print("AUG_DS LOOKS LIKE:", subvolume.shape, label.shape)

del train_ds

"""def get_model(input_shape):
    inputs = keras.Input(input_shape)
    
    x = inputs
    
    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(64, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model"""
# important note: the above model will be ~2 million params!
# it also appears to be using an addition-based residual u-net. I wonder if
# this can be done better using concatenation/densenets, either with conv
# layers or perhaps with self-attention? This is where I want to experiment!
#
# new model will be located in architecture.py!

del volume
del mask
del labels

model = get_model((BUFFER * 2, BUFFER * 2, Z_DIM))
model.summary()
model.compile(
    #optimizer="adam",
    optimizer=tf.keras.optimizers.Adam(learning_rate=learnRate, beta_1=momentum),
    #optimizer=tf.train.experimental.enable_mixed_precision_graph_rewrite(
    #    tf.keras.optimizers.Adam(learning_rate=learnRate, beta_1=momentum)
    #),
    #loss="binary_crossentropy",
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=["accuracy"],
    jit_compile=USE_JIT_COMPILE,
    run_eagerly=True,
)

# for no reason at all, I'll note that this is the last line in monsieur
# Chollet's training regime:
# loss: 0.1243 - accuracy: 0.9513 - val_loss: 0.8680 - val_accuracy: 0.7115
# consider this the number to beat :)

class EveryKCallback(keras.callbacks.Callback):
    def __init__(self,epoch_interval=epoch_interval):
        self.epoch_interval = epoch_interval
    def on_epoch_begin(self,epoch,logs=None):
        if ((epoch % self.epoch_interval)==0):
            self.model.save_weights("ckpts/ckpt"+str(epoch), overwrite=True, save_format='h5')
            #self.model.save('network',overwrite=True)

model.fit(augmented_train_ds,
          validation_data=val_ds,
          epochs=epochs,
          callbacks=[EveryKCallback(epoch_interval=2)], # custom callbacks here!
          #steps_per_epoch=steps_per_epoch
)
model.save("model.keras")

'''def model_train_data(data, label):
    data = model(data)
    return data, label

nan_ds = augmented_train_ds.map(
    model_train_data, num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

for subvolume, label in nan_ds.take(1):
    #print("NANs in subvolume?", tf.math.reduce_any(tf.math.is_nan(subvolume)))
    #print("NANs in label?", tf.math.reduce_any(tf.math.is_nan(label)))
    print("NAN_DS LOOKS LIKE:", subvolume.shape, label.shape)'''

#del volume
#del mask
#del labels
#del train_ds
del augmented_train_ds
del val_ds

# Manually trigger garbage collection
keras.backend.clear_session()
gc.collect()

model = keras.models.load_model("model.keras")


def compute_predictions_map(split, index):
    print(f"Load data for {split}/{index}")

    test_volume = load_volume(split=split, index=index)
    test_mask = load_mask(split=split, index=index)

    test_locations = []
    stride = BUFFER // 2
    for x in range(BUFFER, test_volume.shape[0] - BUFFER, stride):
        for y in range(BUFFER, test_volume.shape[1] - BUFFER, stride):
            test_locations.append((x, y))

    print(f"{len(test_locations)} test locations (before filtering by mask)")

    sample_random_location_test = lambda x: sample_random_location(test_mask.shape)
    is_in_mask_test = lambda x: is_in_masked_zone(x, test_mask)
    extract_subvolume_test = lambda x: extract_subvolume(x, test_volume)

    test_locations_ds = tf.data.Dataset.from_tensor_slices(test_locations).filter(
        is_in_mask_test
    )
    test_ds = test_locations_ds.map(
        extract_subvolume_test, num_parallel_calls=tf.data.AUTOTUNE
    )

    predictions_map = np.zeros(test_volume.shape[:2] + (1,), dtype="float32")
    predictions_map_counts = np.zeros(test_volume.shape[:2] + (1,), dtype="int8")

    print(f"Compute predictions")

    for loc_batch, patch_batch in tqdm(
        zip(test_locations_ds.batch(BATCH_SIZE), test_ds.batch(BATCH_SIZE))
    ):
        predictions = model.predict_on_batch(patch_batch)
        for (x, y), pred in zip(loc_batch, predictions):
            predictions_map[x - BUFFER : x + BUFFER, y - BUFFER : y + BUFFER, :] += pred
            predictions_map_counts[
                x - BUFFER : x + BUFFER, y - BUFFER : y + BUFFER, :
            ] += 1
    predictions_map /= predictions_map_counts + 1e-7
    return predictions_map


predictions_map_a = compute_predictions_map(split="test", index="a")
predictions_map_b = compute_predictions_map(split="test", index="b")

from skimage.transform import resize as resize_ski

original_size_a = PIL.Image.open(DATA_DIR + "/test/a/mask.png").size
predictions_map_a = resize_ski(predictions_map_a, original_size_a).squeeze()

original_size_b = PIL.Image.open(DATA_DIR + "/test/b/mask.png").size
predictions_map_b = resize_ski(predictions_map_b, original_size_b).squeeze()


def rle(predictions_map, threshold):
    flat_img = predictions_map.flatten()
    flat_img = np.where(flat_img > threshold, 1, 0).astype(np.uint8)

    starts = np.array((flat_img[:-1] == 0) & (flat_img[1:] == 1))
    ends = np.array((flat_img[:-1] == 1) & (flat_img[1:] == 0))
    starts_ix = np.where(starts)[0] + 2
    ends_ix = np.where(ends)[0] + 2
    lengths = ends_ix - starts_ix
    return " ".join(map(str, sum(zip(starts_ix, lengths), ())))


threshold = 0.5

rle_a = rle(predictions_map_a, threshold=threshold)
rle_b = rle(predictions_map_b, threshold=threshold)
print("Id,Predicted\na," + rle_a + "\nb," + rle_b, file=open("submission.csv", "w"))
