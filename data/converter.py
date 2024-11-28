# Read the newly uploaded CSV file and convert it into a plain text file
# Skip the first row and separate columns with spaces and rows with newlines.
import csv

csv_file_path = 'Projects\\Image_Classification_in_C\\data\\fashion-mnist_train.csv'
txt_file_output = 'Projects\\Image_Classification_in_C\\data\\fashion-mnist_train.txt'

# Read the CSV file, process it, and write the output to a text file
with open(csv_file_path, 'r') as csv_file:
    csv_reader = list(csv.reader(csv_file))  # Convert to a list to count rows and columns
    num_rows = len(csv_reader) - 1  # Exclude the header row
    num_cols = len(csv_reader[0]) if num_rows > 0 else 0  # Assume consistent columns

# Write the data to the output file
with open(txt_file_output, 'w') as txt_file:
    # Write the number of rows and columns
    txt_file.write(f"{num_rows} {num_cols}\n")
    # Write the rest of the data
    for i, row in enumerate(csv_reader):
        if i == 0:  # Skip the header row
            continue
        txt_file.write(' '.join(row) + '\n')



txt_file_output
