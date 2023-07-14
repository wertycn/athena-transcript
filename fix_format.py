import re
import os

def convert_format(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    corrected_lines = []
    patterns = [
        r"从(.+)导入(.+)",
    ]
    for line in lines:
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                corrected_line = f"import {match.group(2)} from '{match.group(1)}'\n"
                corrected_lines.append(corrected_line)
                break
        else:
            corrected_lines.append(line)

    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(corrected_lines)

def process_directory(directory_path):
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.md'):
                convert_format(os.path.join(root, file))

# 使用脚本


if __name__ == '__main__':

# 使用脚本
    process_directory('D:/mycode/weaviate-docs-zh-main/weaviate-docs-zh-main/developers/weaviate-zh')
