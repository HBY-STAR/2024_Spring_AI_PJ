def process_file(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()
    
    processed_lines = []
    for line in lines:
        # 分割每一行的数据
        index, value = line.strip().split(',')
        # 将字符串转换为浮点数，检查是否小于5
        value = float(value)
        if value < 5:
            value = 5.00
        # 重新格式化字符串，保留两位小数
        processed_line = f"{index},{value:.2f}"
        processed_lines.append(processed_line)
    
    # 将处理后的内容写回文件
    with open(output_file, 'w') as file:
        file.write('\n'.join(processed_lines))

# 示例调用
input_file = 'ansbest2.txt'
output_file = 'processed_numbers.txt'
process_file(input_file, output_file)
