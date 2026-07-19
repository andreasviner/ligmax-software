import os
import shutil

# Define the paths
base_dir = 'filepath/navier_dataset30k'  # Replace with your base directory path 
images_dir = os.path.join(base_dir, 'images')
labels_dir = os.path.join(base_dir, 'labels')

# Create images and labels directories if they don't exist
os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

# Define the subdirectories
subdirs = ['test', 'train', 'valid']

for subdir in subdirs:
    subdir_path = os.path.join(base_dir, subdir)
    images_subdir_path = os.path.join(subdir_path, 'images')
    labels_subdir_path = os.path.join(subdir_path, 'labels')
    
    # Move image files
    if os.path.exists(images_subdir_path):
        for file_name in os.listdir(images_subdir_path):
            src_file = os.path.join(images_subdir_path, file_name)
            dst_file = os.path.join(images_dir, file_name)
            print(f"Moving image file {src_file} to {dst_file}")
            shutil.move(src_file, dst_file)
        
    # Move label files
    if os.path.exists(labels_subdir_path):
        for file_name in os.listdir(labels_subdir_path):
            src_file = os.path.join(labels_subdir_path, file_name)
            dst_file = os.path.join(labels_dir, file_name)
            print(f"Moving label file {src_file} to {dst_file}")
            shutil.move(src_file, dst_file)

print("Files have been successfully moved.")
