from time import sleep
def add_space_between_chars(s):
    return ' '.join(list(s))

def standardization(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'w',
                                                                      encoding='utf-8') as outfile:
        for line in infile:
            line = line.strip()  # 去除原始行的换行符
            middle_index = len(line) // 2

            first_part = add_space_between_chars(line[:middle_index])
            second_part = add_space_between_chars(line[middle_index:])

            new_line = first_part + "\t" + second_part
            outfile.write(new_line + "\n")


standardization('../data/devided.txt','../data/Finaldevided.txt')


