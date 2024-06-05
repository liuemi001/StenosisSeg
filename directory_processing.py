import os

def rename_files(directory_path):
    files = [f for f in os.listdir(directory_path) if f.endswith(".bmp")]
    files.sort()
    for index, filename in enumerate(files):
        new_filename = f"{index + 1}.bmp"
        old_file = os.path.join(directory_path, filename)
        new_file = os.path.join(directory_path, new_filename)
        os.rename(old_file, new_file)
        print(f'Renamed: {old_file} -> {new_file}')

rename_files('/srv/danilov/dataset/')