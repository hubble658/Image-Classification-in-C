
import matplotlib.pyplot as plt

# Dosya yolunu buraya girin
file_path = "D:\Codes\Projects\Image_Classification_in_C\son\\10_Sınıflı\\adam.txt"

# Verileri saklamak için boş listeler oluştur
loss = []
time = []
iteration = []

# Dosyayı oku ve verileri ayır
with open(file_path, "r") as file:
    for line in file:
        try:
            # Her satırı ayrıştır
            l, t, i = map(float, line.strip().split())
            loss.append(l)
            time.append(t)
            iteration.append(i)
        except ValueError:
            print(f"Satır işlenemedi: {line.strip()}")

# Grafik oluşturma
plt.figure(figsize=(12, 6))

# Loss-Time Grafiği
plt.subplot(1, 2, 1)
plt.plot(time, loss, label="Loss-Time", color="blue", marker="o", linewidth=0.5, markersize=4)
plt.xlabel("Time(seconds)")
plt.ylabel("Loss")
plt.title("Loss vs Time")
plt.grid()
plt.legend()

# Loss-Iteration Grafiği
plt.subplot(1, 2, 2)
plt.plot(iteration, loss, label="Loss-Iteration", color="green", marker="o", linewidth=0.5, markersize=4)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss vs Iteration")
plt.grid()
plt.legend()

# Grafikleri göster
plt.tight_layout()
plt.show()
