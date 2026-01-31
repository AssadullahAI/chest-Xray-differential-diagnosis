import tensorflow as tf
import pandas as pd
import numpy as np
import os

# =====================
# CONFIG
# =====================
CSV_PATH = "data_multilabel/labels.csv"
IMAGE_DIR = "data_multilabel/images"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16      # small to avoid OOM
EPOCHS = 5

CHECKPOINT_PATH = "checkpoint_weights.h5"
MODEL_PATH = "xray_multilabel_model.keras"

DISEASES = [
    "Cardiomegaly",
    "Edema",
    "Effusion",
    "Pneumonia",
    "Atelectasis",
    "Consolidation",
    "No Finding"
]

# =====================
# LABEL ENCODER
# =====================
def encode_labels(label_string):
    labels = label_string.split("|")
    return np.array([1 if d in labels else 0 for d in DISEASES], dtype=np.float32)

# =====================
# DATASET LOADER
# =====================
def load_dataset(csv_path, image_dir, split="train"):
    df = pd.read_csv(csv_path)

    # shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    total = len(df)
    train_end = int(0.8 * total)
    val_end = int(0.9 * total)

    if split == "train":
        df = df[:train_end]
    elif split == "val":
        df = df[train_end:val_end]
    else:
        df = df[val_end:]

    image_paths = []
    labels = []

    for _, row in df.iterrows():
        img_path = os.path.join(image_dir, row["Image Index"])
        if os.path.exists(img_path):
            image_paths.append(img_path)
            labels.append(encode_labels(row["Finding Labels"]))

    image_paths = np.array(image_paths)
    labels = np.array(labels)

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    def parse_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = img / 255.0
        return img, label

    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)

    if split == "train":
        dataset = dataset.shuffle(5000)

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

# =====================
# CREATE DATASETS
# =====================
train_ds = load_dataset(CSV_PATH, IMAGE_DIR, split="train")
val_ds = load_dataset(CSV_PATH, IMAGE_DIR, split="val")

num_classes = len(DISEASES)

# =====================
# MODEL
# =====================
base_model = tf.keras.applications.EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3),
)

base_model.trainable = False

x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.Dense(512, activation="relu")(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)

model = tf.keras.Model(base_model.input, outputs)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# =====================
# CHECKPOINT
# =====================
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    CHECKPOINT_PATH,
    save_best_only=False,
    save_weights_only=True
)

# =====================
# RESUME IF CHECKPOINT EXISTS
# =====================
if os.path.exists(CHECKPOINT_PATH):
    print("⚠️  Checkpoint found. Resuming training...")
    model.load_weights(CHECKPOINT_PATH)

# =====================
# TRAIN
# =====================
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb]
)

# =====================
# SAVE FINAL MODEL
# =====================
model.save(MODEL_PATH)
print("✅ Training complete and model saved!")
