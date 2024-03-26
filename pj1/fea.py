from Bio.PDB import PDBParser
import os
import numpy as np


# https://biopython.org/docs/1.75/api/Bio.PDB.Atom.html

def feature_extraction():
    parser = PDBParser(PERMISSIVE=1, QUIET=True)
    all_atom_names = set()
    file_names = []

    # 第一步：遍历一次以收集所有不同的原子名称
    for file in os.listdir("./data/SCOP40mini"):
        structure_id = os.path.splitext(file)[0]
        structure = parser.get_structure(structure_id, "./data/SCOP40mini/" + file)
        file_names.append(file)
        for atom in structure.get_atoms():
            all_atom_names.add(atom.get_name())

    # 创建一个列表，按照排序的原子名称
    atom_names_sorted = sorted(list(all_atom_names))

    print("Atom names:", atom_names_sorted)

    # 初始化一个空的NumPy矩阵
    atom_matrix = np.zeros((len(file_names), len(atom_names_sorted)))

    # 第二步：再次遍历每个文件，更新矩阵中相应的原子数量
    for file_index, file in enumerate(os.listdir("./data/SCOP40mini")):
        structure_id = os.path.splitext(file)[0]
        structure = parser.get_structure(structure_id, "./data/SCOP40mini/" + file)
        atom_counts = dict.fromkeys(atom_names_sorted, 0)

        for atom in structure.get_atoms():
            atom_counts[atom.get_name()] += 1

        # 更新矩阵
        for atom_index, atom_name in enumerate(atom_names_sorted):
            atom_matrix[file_index, atom_index] = atom_counts[atom_name]

        if (file_index + 1) % 100 == 0:
            print(f"Processed file {file_index + 1} of {len(file_names)}")

    print("Feature extraction completed.")  # 打印函数完成消息

    return atom_matrix, atom_names_sorted


if __name__ == "__main__":
    # 调用函数并获取返回的矩阵和原子名称列表
    atom_matrix, atom_names_sorted = feature_extraction()

    # 打印结果以验证
    print("Atom names:", atom_names_sorted)
    print("Matrix shape:", atom_matrix.shape)

    # 打印矩阵的一小部分或特定行列以查看具体值
