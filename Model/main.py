import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Ustawienie wielkości obrazów i batcha
image_size = (369, 369)
batch_size = 20

# Przydzielenie 80% danych do zbioru treningowego
train_1 = keras.utils.image_dataset_from_directory(
    'Gesty',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=image_size,
    batch_size=batch_size
)

# Wzięcie nazw klas ze zbioru treningowego
class_names = train_1.class_names

# Augmentacja danych, losowy obrót do 18 stopni i przesuniecie do ok 19 pikseli poziomo/pionowo
data_augmentation = keras.Sequential([
    layers.RandomRotation(0.05),
    layers.RandomTranslation(0.05, 0.05),
])

# Metoda autotune do optymalziacji ładowania
AUTOTUNE = tf.data.AUTOTUNE

# Użycie augmentacji i autotune na danych
train = train_1.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=AUTOTUNE
).cache().prefetch(AUTOTUNE)

# Przypisanie pozostałych 20% danych do zbioru walidacyjnego
val = keras.utils.image_dataset_from_directory(
    'Gesty',
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=image_size,
    batch_size=batch_size
)

# Użycie cache i prefetch do szybszego przebiegu treningu
train = train.cache().prefetch(buffer_size=AUTOTUNE)
val = val.cache().prefetch(buffer_size=AUTOTUNE)

'''
Definicja modelu ML z CNN
warstwa wejściowa
warstwa Rescaling do zamiany wartości z 0-255 do 0-1
warstwy Conv2D do wykrywanbia cech obrazów
warstwy MaxPooling do redukcji cech wymiarów na fragmenty 2x2 poprzez wybór największej wartości
warstwa GlobalAveragePooling do globalenej redukcji przez uśrednianie
warstwa Dropout do losowego ustawiania wartości neuronów na 0, żeby sieć się nie przeuczała
warstwa wyjściowa
'''


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


# Utworzenie modelu z zadanym kształtem wejściowym, liczbą klas i ustawianiem wartości dropout
model = make_model(input_shape=image_size + (3,),
                   num_classes=len(class_names), dropout_rate=0)
# Kompilacja modelu z optymalizatorem Adam, funkcją straty i metryką dokładności
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Ustawienie liczby epok
epochs = 20

# Zmienna, która przechowuje wyniki dokładności i funkcji straty
history = model.fit(
    train,
    epochs=epochs,
    validation_data=val
)


# Wykres przedstawiający dokładność modelu
def plot_accuracy(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("Dokładność modelu")
    plt.ylabel("Dokładność")
    plt.xlabel("Epoka")
    plt.legend(["Dokładność treningowa", "Dokładność walidacyjna"])
    plt.show()


# Wykres przedstawiający funkcję straty modelu
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

# Pobranie danych ze zbioru walidacyjnego
# Predykcja etykiet przez model
imgs, labs = next(iter(val))
preds = model.predict(imgs).argmax(axis=1)

# Wybranie po n_samples obrazów dla każdej klasy
n_classes = len(class_names)
n_samples = 2

# Utworzenie siatki wykresów do wizualizacji i zamiana do numpy
fig, axes = plt.subplots(n_samples, n_classes,
                         figsize=(n_classes*3, n_samples*3))
labs_np = labs.numpy()

# Iteracja po klasach i wybór n_samples przykładów z każdej
for clas in range(n_classes):
    ids = np.flatnonzero(labs_np == clas)[:n_samples]
    for i, ax in enumerate(axes[:, clas]):
        # Wyświetlenie obrazu z etykietą rzeczywistą (T) i przewidywaną (P)
        ax.imshow(imgs[ids[i]].numpy().astype('uint8'))
        ax.set_title(f"T:{class_names[clas]}\nP:{class_names[preds[ids[i]]]}")
        ax.axis('off')

plt.tight_layout()
plt.show()

# Zebranie wszystkich obrazów i etykiet ze zbioru walidacyjnego
all_imgs = []
all_labels = []
for batch_imgs, batch_labels in val:
    all_imgs.append(batch_imgs)
    all_labels.append(batch_labels)

# Połączenie batchy w jedną tablicę
all_imgs = tf.concat(all_imgs, axis=0)
all_labels = tf.concat(all_labels, axis=0)

# Przewidywanie etykiet dla całego zbioru walidacyjnego
all_preds = model.predict(all_imgs).argmax(axis=1)

# Wyświetlenie macierzy pomylek
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Macierz pomyłek")
plt.tight_layout()
plt.show()
