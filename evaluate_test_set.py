# Author: Jacob Dawson
#
# splitting this into its own script both for logical reasons...and because
# I keep getting OOM errors when this code is after the train code.

from script import *
from tqdm import tqdm
from skimage.transform import resize as resize_ski

# Let's do predictions!
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

gc.collect()
predictions_map_a = compute_predictions_map(split="test", index="a")
gc.collect()
predictions_map_b = compute_predictions_map(split="test", index="b")
gc.collect()

original_size_a = PIL.Image.open(DATA_DIR + "/test/a/mask.png").size
predictions_map_a = resize_ski(predictions_map_a, original_size_a).squeeze()

gc.collect()

original_size_b = PIL.Image.open(DATA_DIR + "/test/b/mask.png").size
predictions_map_b = resize_ski(predictions_map_b, original_size_b).squeeze()

gc.collect()

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

gc.collect()

rle_a = rle(predictions_map_a, threshold=threshold)
gc.collect()

rle_b = rle(predictions_map_b, threshold=threshold)
gc.collect()

print("Id,Predicted\na," + rle_a + "\nb," + rle_b, file=open("submission.csv", "w"))
