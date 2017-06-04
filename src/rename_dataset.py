import os
from src import myUtil


def rename_dataset(inputs_path, teachers_path):
    files = os.listdir(inputs_path)
    file_index = 0
    for f in files:
        _, ext = os.path.splitext(f)
        input_file_path = os.path.join(inputs_path, f)
        if not myUtil.is_img(input_file_path):
            continue
        teacher_file_path = os.path.join(teachers_path, f)
        if not os.path.exists(teacher_file_path):
            print("teacher data of " + f + " don't exist.")
            continue
        #ここから
        input_dst = os.path.join(inputs_path, str(file_index)) + ext
        teacher_dst = os.path.join(teachers_path, str(file_index)) + ext
        os.rename(input_file_path, input_dst)
        os.rename(teacher_file_path, teacher_dst)
        file_index += 1

if __name__ == '__main__':
    rename_dataset("./imgs/patches/contour", "./imgs/patches/pencil")
