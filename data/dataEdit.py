input_file = "Projects/Image_Classification_in_C/data/train_convert.txt"
output_file = "Projects/Image_Classification_in_C/data/train_0_1.txt"


with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    lines = infile.readlines()
    
    # Write the first line unchanged
    outfile.write(lines[0])
    
    # Process the rest of the lines
    for line in lines[1:]:
        numbers = line.strip().split()
        if(int(numbers[0])== 0 or int(numbers[0])== 1 ):
            row = [numbers[0]] + [f"{float(num) / 255:.3f}" for num in numbers[1:]]
            outfile.write(" ".join(row) + "\n")
