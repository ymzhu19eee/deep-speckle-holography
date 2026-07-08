
# Deep Speckle Holography

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the official code repository for the paper **"Label-free Single-Nanoparticle Deep Speckle Holography"**. 

This project is from the **Imaging Systems Lab**, Department of Electrical and Computer Engineering, The University of Hong Kong, and the **Tadesse Lab**, Department of Mechanical Engineering, Massachusetts Institute of Technology.

**Corresponding authors:** Edmund Y. Lam, Loza F. Tadesse, Yanmin Zhu  
**Repository:** [https://github.com/ymzhu19eee/deep-speckle-holography](https://github.com/ymzhu19eee/deep-speckle-holography)  
**Detailed Code Functionality:** Please refer to *Supplemental Information Methods and Notes 2* of the manuscript.

---

## 1. System Requirements

### Hardware Requirements
The code requires a standard computer with enough RAM to support the in-memory operations. For optimal performance during model training and inference, a **CUDA-enabled GPU** (e.g., NVIDIA RTX 3090 or equivalent) is highly recommended.

### Software Requirements
This package is supported for *Linux*, *macOS*, and *Windows* operating systems. The code has been tested on the following systems:
- **OS:** Ubuntu 20.04 / Windows 10
- **Python:** 3.7, 3.8 (Tested on 3.8)
- **PyTorch:** 1.10 or higher (Tested on 1.10.0+cu113)

**Dependencies:**
- `torchvision >= 0.11`
- `pandas`
- `scikit-learn`
- `Pillow`

---

## 2. Installation Guide

### Instructions
It is recommended to use `conda` or `pip` to manage your Python environment and install the dependencies.

```bash
# 1. Clone the repository
git clone https://github.com/ymzhu19eee/deep-speckle-holography.git
cd deep-speckle-holography

# 2. Create a virtual environment (recommended)
conda create -n speckle_holo python=3.8
conda activate speckle_holo

# 3. Install dependencies
pip install torch==1.10.0 torchvision==0.11.1 pandas scikit-learn Pillow

**Usage**

*Requirements*： <br />
Python >= 3.7 
PyTorch >= 1.10 
torchvision >= 0.11
pandas 
scikit-learn 
Pillow 

It is recommended to use conda or pip to manage your Python environment and install the dependencies.


**1. Data requirements**

Place merged_output.csv in the same directory as the script
Or manually modify the CSV_PATH field in the script to specify its location.

**2. Prepare the Images**

In merged_output.csv, the image_path column should specify the paths to the images.
The script will automatically read and convert them to RGB, then apply preprocessing (224×224 resize, normalization, etc.).
Run Stage 1 Classification Training


After training, two weight files will be generated in the current directory:
particle_classifier.pt: Encoder model weights


**License**
This project is available under the MIT license.

**Reference**
If you use this code, please provide a citation of the paper.
