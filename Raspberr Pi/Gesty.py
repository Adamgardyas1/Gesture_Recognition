import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from tensorflow import keras
from matplotlib import cm
from vl53l5cx_ctypes import VL53L5CX

# Ustawienie rozmiaru wejsciowego obrazow i maksymalnego dystansu odczytu czujnika
IMAGE_SIZE = (369, 369)
dist = 500.0

# Inicjalizacja czujnika
sensor = VL53L5CX()
sensor.set_resolution(8*8)
sensor.set_ranging_frequency_hz(10)
sensor.start_ranging()

# Uruchomienie trybu interaktywnego wykresu
plt.ion()
fig, ax = plt.subplots()
heatmap = ax.imshow(np.zeros((8, 8)), cmap='Reds', vmin=0,
                    vmax=dist)  # wykres mapy głebokosci
ax.axis('off')
plt.show()

# Wczytanie wytrenowanego modelu i dodanie nazw klas
model = keras.models.load_model('gesty_model.h5')
class_names = ["Dol", "Gora", "Lewo", "Prawo", "Stop"]

# Wczytanie mapy do konwersji na RGB
cmap = cm.get_cmap('Reds')
last_inference = time.time()  # Czas ostatniej predykcji

try:
    while True:
        # Sprawdzenie czy sa nowe dane w czujniku
        if sensor.data_ready():
            data = sensor.get_data()
            distances = np.array(data.distance_mm).reshape((8, 8))
            distances = np.flipud(distances)
            heatmap.set_data(distances)
            plt.pause(0.05)

            # Wykonywanie predykcji co 0.5 sekundy
            now = time.time()
            if now - last_inference >= 0.5:
                last_inference = now
                # Normalizacja wartosci do 0 - 1
                normed = distances / dist
                # Zmiana na obraz RGB
                rgba = cmap(normed)
                # Wybor kanałow RGB i przeskalowanie ich do 0 - 255
                rgb_uint8 = (rgba[..., :3] * 255).astype(np.uint8)
                # Skalowanie obrazu do tego, ktory był uzyty podczas uczenia
                img = tf.image.resize(rgb_uint8.astype(
                    np.float32), IMAGE_SIZE).numpy()
                # Dodanie wymiaru batcha
                input_tensor = np.expand_dims(img, axis=0)
                # Wykonanie predykcji
                preds = model.predict(input_tensor)
                # Wybor klasy z najwiekszym prawdopodobiestwem
                idx = np.argmax(preds, axis=1)[0]
                confidence = preds[0][idx] * 100

                print(f"{class_names[idx]}: {confidence:.2f}%")

        time.sleep(0.01)

# Zatrzymanie czujnika i zamkniecie okna wykresu
finally:
    sensor.stop_ranging()
    plt.ioff()
    plt.show()
