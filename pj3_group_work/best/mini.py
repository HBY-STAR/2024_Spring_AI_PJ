def read_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [line.strip().split(',') for line in lines]

def write_file(file_path, data):
    with open(file_path, 'w') as file:
        for line in data:
            file.write(f"{line[0]},{line[1]}\n")

def compare_files(file1_path, file2_path, output_path):
    data1 = read_file(file1_path)
    data2 = read_file(file2_path)

    min_values = []
    for line1, line2 in zip(data1, data2):
        index1, value1 = line1
        index2, value2 = line2
        if index1 != index2:
            raise ValueError("The indices of the lines do not match.")
        
        min_value = min(float(value1), float(value2))
        min_values.append((index1, min_value))
    
    write_file(output_path, min_values)

# 使用示例
file1_path = 'ansmix2.txt'
file2_path = 'ans_ori.txt'
output_path = 'output.txt'

compare_files(file1_path, file2_path, output_path)
