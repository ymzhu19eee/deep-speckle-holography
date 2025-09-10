**Deep Speckle Holography** <br />
This is the official code repository for the paper "*Unveiling multi-dimensional nanoparticle signatures via deep speckle holography*". This project is from the Imaging Systems Lab, Department of Electrical and Electronic Engineering, The University of Hong Kong.

Overview


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
