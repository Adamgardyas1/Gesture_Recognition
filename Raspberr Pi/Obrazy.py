from vl53l5cx_ctypes import VL53L5CX
import numpy as np
import matplotlib.pyplot as plt
import time

# Inicjalizacaj czujnika VL53L5CX
sensor = VL53L5CX()
sensor.set_resolution(8*8)  # Ustawienie rozdzialczosci na 8x8
sensor.set_ranging_frequency_hz(10)  # Ustawienie czestotliwosci na 10 hz
sensor.start_ranging()  # Rozpoczecie pomiarow

# Wlaczenie interaktywnego trybu wykresow
plt.ion()
fig, ax = plt.subplots()

# Utworzenie pustej mapy z siatka 8x8 i z zakresem od 0 do 500
heatmap = ax.imshow(np.zeros((8, 8)), cmap='Reds', vmin=0, vmax=500)
ax.axis('off')  # Ukrycie osi

# Licznik obrazow
pomiar = 0
running = True

# Petla pomiarowa
while running:
    if sensor.data_ready():
        data = sensor.get_data()
        distances = np.array(data.distance_mm).reshape(
            (8, 8))  # Przeksztalcenie danych do macierzy 8x8
        distances = np.flipud(distances)  # Odwrocenie danych w pionie
        heatmap.set_data(distances)  # Aktualizacja danych na mapie
        plt.draw()
        plt.pause(0.05)  # Opoznienie wyswietlania
        # Zapis mapy jako plik graficzny PNG o domyslnym rozmiarze 369x369 pikseli
        filename = f"Prawo/Prawo_1_{pomiar}.png"
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        pomiar += 1  # Kolejne iteracje
    time.sleep(0.1)  # Opozninie miedzy iteracjami
    if pomiar >= 1000:  # Zatrzymanie po wykonaniu 1000 iteracji
        running = False

# Zatrzymanie dzialania czujnika i wylaczenie tryby interaktywnego wykresu
sensor.stop_ranging()
plt.ioff()
plt.show()
