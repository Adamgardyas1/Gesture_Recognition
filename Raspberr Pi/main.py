# Żeby uruchomić wytrenowany model na innym urządzeniu należy przenieść plik zapisanego modelu jak i skrypt, który go ładuje oraz uruchamia predykcję

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Ustawienia
image_size = (369, 369)
batch_size = 20

train_1 = keras.utils.image_dataset_from_directory(
    'Gesty',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=image_size,
    batch_size=batch_size
)

class_names = train_1.class_names

data_augmentation = keras.Sequential([
    layers.RandomRotation(0.05),
    layers.RandomTranslation(0.05, 0.05),
])

AUTOTUNE = tf.data.AUTOTUNE

train = train_1.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=AUTOTUNE
).cache().prefetch(AUTOTUNE)

val = keras.utils.image_dataset_from_directory(
    'Gesty',
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=image_size,
    batch_size=batch_size
)

train = train.cache().prefetch(buffer_size=AUTOTUNE)
val = val.cache().prefetch(buffer_size=AUTOTUNE)

# Definicja modelu


def make_model(input_shape, num_classes, dropout_rate):
    inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(8, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2, 2)(x)

    x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2, 2)(x)

    x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2, 2)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs)


model = make_model(input_shape=image_size + (3,),
                   num_classes=len(class_names), dropout_rate=0)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

epochs = 15

history = model.fit(
    train,
    epochs=epochs,
    validation_data=val
)


def plot_accuracy(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("Dokładność modelu")
    plt.ylabel("Dokładność")
    plt.xlabel("Epoka")
    plt.legend(["Dokładność treningowa", "Dokładność walidacyjna"])
    plt.show()


def plot_loss(hist):
    plt.plot(hist.history["loss"])
    plt.plot(hist.history["val_loss"])
    plt.title("Strata modelu")
    plt.ylabel("Strata")
    plt.xlabel("Epoka")
    plt.legend(["Strata treningowa", "Strata walidacyjna"])
    plt.show()


plot_accuracy(history)
plot_loss(history)

# Zapis modelu w formacie HDF5
model.save('gesty_model.h5')

imgs, labs = next(iter(val))
preds = model.predict(imgs).argmax(axis=1)

n_classes = len(class_names)
n_samples = 2

fig, axes = plt.subplots(n_samples, n_classes,
                         figsize=(n_classes*3, n_samples*3))
labs_np = labs.numpy()

for clas in range(n_classes):
    ids = np.flatnonzero(labs_np == clas)[:n_samples]
    for i, ax in enumerate(axes[:, clas]):
        ax.imshow(imgs[ids[i]].numpy().astype('uint8'))
        ax.set_title(f"T:{class_names[clas]}\nP:{class_names[preds[ids[i]]]}")
        ax.axis('off')

plt.tight_layout()
plt.show()
