import os

def is_img(path):
    root, ext = os.path.splitext(path)
    return ext[1:] in ['jpg', 'png']

def img_paths(dir_path):
    list_dir = os.listdir(dir_path)
    paths = [os.path.abspath(os.path.join(dir_path, f)) for f in list_dir]
    return [f for f in paths if is_img(f)]
