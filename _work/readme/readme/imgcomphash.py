import os
import sys
import imagehash
from PIL import Image

def find_similar_images(dir1, dir2, hash_size=16, output_file=None):
    # Redirect print statements to the output file if specified
    if output_file:
        sys.stdout = open(output_file, 'w')

    print("Comparing images between two directories")
    
    # Get image files from both directories
    image_filenames1 = [(f, dir1) for f in os.listdir(dir1) if f.endswith(('jpg', 'jpeg', 'png', 'bmp'))]
    image_filenames2 = [(f, dir2) for f in os.listdir(dir2) if f.endswith(('jpg', 'jpeg', 'png', 'bmp'))]
    
    image_filenames = image_filenames1 + image_filenames2
    print(f"Found {len(image_filenames1)} images in {dir1} and {len(image_filenames2)} images in {dir2}")
    
    hashes = {}
    for filename, directory in image_filenames:
        try:
            print(f"Processing file: {filename} in directory: {directory}")
            img = Image.open(os.path.join(directory, filename))
            img_hash = imagehash.average_hash(img, hash_size=hash_size)
            if img_hash in hashes:
                print(f"Found similar images: {filename} in {directory} and {hashes[img_hash][0]} in {hashes[img_hash][1]}")
                print(f"Hash: {img_hash}")
            else:
                hashes[img_hash] = (filename, directory)
        except Exception as e:
            print(f"Error processing file {filename} in directory {directory}: {e}")

    # Restore the original stdout
    if output_file:
        sys.stdout.close()
        sys.stdout = sys.__stdout__

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 imgcomphash.py <folder_path1> <folder_path2> <output_file>")
        sys.exit(1)

    folder_path1 = sys.argv[1]
    folder_path2 = sys.argv[2]
    output_file = sys.argv[3]

    if not os.path.isdir(folder_path1):
        print(f"Error: {folder_path1} is not a valid directory")
        sys.exit(1)
    
    if not os.path.isdir(folder_path2):
        print(f"Error: {folder_path2} is not a valid directory")
        sys.exit(1)

    find_similar_images(folder_path1, folder_path2, output_file=output_file)
