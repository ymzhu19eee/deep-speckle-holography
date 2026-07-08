# Deep Speckle Holography

[![License: MIT][image_0]](https://opensource.org/licenses/MIT)

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
```

---

## 3. Demo

To verify the installation, you can run a quick demonstration using the provided sample data with `https://drive.google.com/drive/folders/1QmjhYn_9KDOeBymzpkbAwQJEkM0H3L0L?usp=sharing`.

**Instructions:**

```bash
python train.py --csv_path demo_data/merged_output.csv
```
*(Note: Please replace `train.py` and `demo_data/` with your actual script and demo folder names if different).*

- **Expected Output:** The script will output training loss logs in the terminal. After completion, `particle_classifier.pt` will be generated in the current directory.
- **Expected Run Time:** Approximately 5 minutes on a modern GPU.

---

## 4. Instructions for Use

To run the software on your own data, please follow the steps below:

### 4.1 Data Requirements
Place `merged_output.csv` in the same directory as the script. Alternatively, you can manually modify the `CSV_PATH` field in the script to specify its location.

### 4.2 Prepare the Images
In `merged_output.csv`, the `image_path` column should specify the paths to the images. The script will automatically read and convert them to RGB, then apply preprocessing (224×224 resize, normalization, etc.).

### 4.3 Run Stage 1 Classification Training
Execute the training script:

```bash
python train.py
```

After training, two weight files will be generated in the current directory:
- `particle_classifier.pt`: Encoder model weights
- `classifier_head.pt`

### 4.4 Reproduction Instructions
To reproduce all the quantitative results in the manuscript, please download the full dataset from the Supplemental data of the paper and run the training scripts using the default hyperparameters as configured in this repository.

---

## License
This project is available under the [MIT License](LICENSE).

## Reference
If you use this code, please provide a citation of our paper.

[image_0]: https://pfst.cf2.poecdn.net/base/image/0f22612a092cd1947cd99e9a61a24fc5c955c4a2986d89bc0991dfcd0a8d42f2?pmaid=635822161
