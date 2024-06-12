def read_file(file_path):
    data = {}
    with open(file_path, 'r') as file:
        for line in file:
            position, value = line.strip().split(',')
            data[int(position)] = float(value)
    return data

def calculate_differences(file1_data, file2_data):
    differences = {}
    for position in file1_data:
        if position in file2_data:
            differences[position] = file1_data[position] - file2_data[position]
            differences[position] = 0 if abs(differences[position]) < 1 else differences[position]
        else:
            differences[position] = file1_data[position]
            print(f"Position {position} is not in file2")
    for position in file2_data:
        if position not in differences:
            differences[position] = -file2_data[position]
            print(f"Position {position} is not in file1")
    return differences

def write_differences_to_file(differences, output_file_path):
    with open(output_file_path, 'w') as file:
        for position in sorted(differences.keys()):
            file.write(f"{position},{differences[position]:.2f}\n")

def main():
    file1_path = 'ansbest.txt'
    file2_path = 'ans445.txt'
    output_file_path = 'difference.txt'

    file1_data = read_file(file1_path)
    file2_data = read_file(file2_path)
    differences = calculate_differences(file1_data, file2_data)
    write_differences_to_file(differences, output_file_path)

if __name__ == "__main__":
    main()
