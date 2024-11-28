import matplotlib.pyplot as plt

# File name
file_name = "data.txt"

# Initialize lists to store the data
loss_values = []
iterations = []
times = []

# Read the data from the file
with open(file_name, 'r') as file:
    for line in file:
        try:
            # Split the line into components
            loss, iteration, time = map(float, line.split())
            loss_values.append(loss)
            iterations.append(iteration)
            times.append(time)
        except ValueError:
            # Skip lines that don't match the expected format
            print(f"Skipping invalid line: {line.strip()}")

# Plot Loss vs Time
plt.figure(figsize=(10, 5))
plt.plot(times, loss_values, marker='o', linestyle='-', color='b')
plt.title("Loss vs Time")
plt.xlabel("Time")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# Plot Loss vs Iteration
plt.figure(figsize=(10, 5))
plt.plot(iterations, loss_values, marker='o', linestyle='-', color='g')
plt.title("Loss vs Iteration")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
