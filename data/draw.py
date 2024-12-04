import numpy as np
import matplotlib.pyplot as plt

def plot_squares_together(matrices, ncols=3):
    """
    Birden fazla kare matris için hepsini aynı anda görselleştirir.
    - matrices: Liste içinde numpy arrayleri.
    - ncols: Alt grafiklerde sütun sayısı.
    """
    n = len(matrices)  # Toplam matris sayısı
    nrows = (n + ncols - 1) // ncols  # Gerekli satır sayısını hesapla
    
    # Figure ve alt grafik düzenini oluştur
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axes = axes.flatten()  # Axisleri 1D array'e çevir
    
    for i, matrix in enumerate(matrices):
        axes[i].imshow(matrix, cmap='gray', interpolation='nearest')
        axes[i].axis('off')
        axes[i].set_title(f"Matrix {i + 1}")
    
    # Kalan boş alt grafiklerin eksenlerini kapat
    for j in range(len(matrices), len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()

def read_and_plot_together(path):
    """
    Dosyadan tam kare boyutundaki matrisleri okuyup hepsini aynı anda çizdirir.
    """
    matrices = []  # Kare matrisleri saklamak için bir liste
    
    # Dosyayı aç ve satır satır oku
    with open(path, 'r') as file:
        lines = file.readlines()
    
    for line_num, line in enumerate(lines):
        # Satır içeriğini float listeye dönüştür
        input_array = list(map(float, line.split()))
        
        # Eğer tam kare ise listeye ekle
        n = len(input_array)
        side_length = int(np.sqrt(n))
        if side_length ** 2 == n:
            square_matrix = np.array(input_array).reshape((side_length, side_length))
            matrices.append(square_matrix)
        else:
            print(f"Line {line_num + 1} skipped: Not a perfect square")
    
    # Eğer herhangi bir matris varsa görselleştir
    if matrices:
        plot_squares_together(matrices)
    else:
        print("No valid matrices to plot.")

# Example usage
path = "D:/Codes/Projects/Image_Classification_in_C/src/filtered.txt"
read_and_plot_together(path)
