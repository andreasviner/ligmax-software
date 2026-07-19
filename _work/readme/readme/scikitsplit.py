import os
import shutil
import glob
from sklearn.model_selection import train_test_split

# specify the path
image_path = "filepath/images"
label_path = "filepath/labels"

# Define your new directories for images and labels
new_dir_paths = ['filepath/dataset/train',
                 'filepath/dataset/valid',
                 'filepath/dataset/test']

# Get list of all .png and .jpg files
image_files = glob.glob(os.path.join(image_path, "*.png")) + glob.glob(os.path.join(image_path, "*.jpg"))
label_files = [os.path.join(label_path, os.path.basename(f).rsplit('.', 1)[0] + '.txt') for f in image_files]

print(len(image_files), len(label_files))

# Split data into training + validation (98%) and test (2%)
train_images, temp_images, train_labels, temp_labels = train_test_split(image_files, label_files, test_size=0.02, random_state=42)

# Split the training + validation into train (76.53%) and validation (23.47%)
valid_images, test_images, valid_labels, test_labels = train_test_split(temp_images, temp_labels, test_size=0.2347, random_state=42)

# Create separate directories for images and labels in each split
for path in new_dir_paths:
    os.makedirs(os.path.join(path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(path, 'labels'), exist_ok=True)

# Now, you can copy your files to the respective directories
for image, label in zip(train_images, train_labels):
    shutil.copy(image, os.path.join(new_dir_paths[0], 'images'))
    if os.path.exists(label):
        shutil.copy(label, os.path.join(new_dir_paths[0], 'labels'))

for image, label in zip(valid_images, valid_labels):
    shutil.copy(image, os.path.join(new_dir_paths[1], 'images'))
    if os.path.exists(label):
        shutil.copy(label, os.path.join(new_dir_paths[1], 'labels'))

for image, label in zip(test_images, test_labels):
    shutil.copy(image, os.path.join(new_dir_paths[2], 'images'))
    if os.path.exists(label):
        shutil.copy(label, os.path.join(new_dir_paths[2], 'labels'))