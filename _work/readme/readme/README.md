# Navier Njord Buoy Dataset 2024

## Overview
This folder contains some details and information about the dataset, along with utility scripts for dataset management. The dataset includes 30,157 images and labels distributed across train, test, and valid folders for Njord buoys.

## Author
**Name:** Don Bullecer

**Last Updated:** 7.10.2024

## Dataset Details
`navier_dataset30k.7z` contains Navier's dataset used in the Njord 2024 Competition.

**Source:** Field testing months before the competition and during the testing week. Open source Roboflow Universe.

**Content:** 
  - images and labels for Njord buoys in `train`, `test`, and `valid`
  - `data.yaml`:
  ```
  train: ./train
  val: ./valid
  test: ./test

  nc: 6
  names: ['green','red','north','east','south','west']
  ```
**Special Notes:** 
  - Labels were verified with specific criteria, focusing on correctly labelled buoys of a certain size
  - Small or distant buoys may not be labelled
  - A filter based on pixel area was applied

## Acknowledgements
A huge thank you to the Navier team, and especially to the members who assisted in label verification

## Data Cleaning
- Duplicate images were removed using image hash comparison
- The `imgcomphash.py` script (described below) was used for this process

## Utility Scripts

### 1. scikitsplit.py
**Purpose:** Splits images and labels into test, train, and validation folders.

**Features:**
- Splits data into train (75%), validation (23%), and test (2%) sets
- Creates separate directories for each split
- Uses scikit-learn's train_test_split for randomization

**Usage:**
```
python scikitsplit.py
```

### 2. undoscikitsplit.py
**Purpose:** Reverses the split operation performed by scikitsplit.py.

**Features:**
- Consolidates split data back into single 'images' and 'labels' directories
- Ensures no data loss during reorganization

**Usage:**
```
python undoscikitsplit.py
```

### 3. imgcomphash.py
**Purpose:** Compares images based on perceptual hash values.

**Features:**
- Identifies similar images across directories
- Adjustable hash size for precision control
- Supports various image formats

**Usage:**
```
python imgcomphash.py <folder_path1> <folder_path2> <output_file> [hash_size]
```

## Requirements
- Python 3.x
- scikit-learn
- numpy
- Pillow (PIL)
- imagehash

Install dependencies:
```
pip install scikit-learn numpy Pillow imagehash
```

## Usage Notes and Warnings
1. Always backup data before running scripts
2. Verify file paths and permissions
3. Be aware of potentially long processing times for large datasets
4. Adjust hash size in imgcomphash.py to balance precision and performance

For detailed usage instructions and warnings, refer to the individual script files.

---

Feel free to reach out for any questions or clarifications about the dataset or scripts. (kontakt@navierusn.no)