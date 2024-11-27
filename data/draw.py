import numpy as np
import matplotlib.pyplot as plt

def plot_square(input_array):
    # Ensure the input is a numpy array
    input_array = np.array(input_array)
    
    # Get the size of the array
    n = input_array.size
    
    # Check if n is a perfect square
    side_length = int(np.sqrt(n))
    if side_length ** 2 != n:
        raise ValueError("Input array size must be a perfect square.")
    
    # Reshape the 1*n array to a square matrix (side_length x side_length)
    square_matrix = input_array.reshape((side_length, side_length))
    
    # Plot the matrix
    plt.imshow(square_matrix, cmap='gray', interpolation='nearest')
    
    # Draw the border by adjusting the axis limits and adding a rectangle
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    
    
    
    # Show the plot
    # plt.axis('off')  # Optional: turn off axes for cleaner visualization
    plt.show()

# Example usage
input_array = list(map(float,input().split() )) # Replace this with your own 1*n array

plot_square(input_array)
