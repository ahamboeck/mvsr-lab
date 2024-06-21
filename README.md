# Overview
MVSR Lab is a project focused on [provide a brief description of what the project is about]. This project aims to [mention the main goals or objectives of the project].

## Directory Structure
Data
data/test
WhatsApp Video 2024-05-19 ...: Video files used for model testing.
test_video.mp4: A test video file.


data/train
outputs: Directory for output files generated during training.
snippets: Directory containing snippets related to training data.
train_images: Directory containing images used for training.

Demo
demo_vid.mp4: A demonstration video file.


Models
confusion_matrices: Directory containing confusion matrices.
mcnemar_test_results: Directory for results from McNemar's test.
model_dropout: Directory containing models with dropout applied.
model_dropout_2_conv: Directory containing models with dropout and 2 convolution layers.
model_no_dropout: Directory containing models without dropout.
model_no_dropout_2_conv: Directory containing models without dropout and 2 convolution layers.


Scripts
detect_markers.ipynb: Jupyter notebook for marker detection.
detect_markers.py: Python script for marker detection.
prepare_markers.ipynb: Jupyter notebook for preparing markers.
requirements.txt: List of dependencies required for the project.
train_model.ipynb: Jupyter notebook for training the model.


Root Directory
.gitattributes: Git attributes file.
.gitignore: Git ignore file to exclude specific files and directories from version control.
flowchart.drawio: A file likely containing a flowchart diagram.
get-pip.py: A script to install pip, the Python package installer.
Installation
To install and set up this project locally, follow these steps:

Clone the repository:
sh
Code kopieren
git clone https://github.com/ahamboeck/mvsr-lab.git
Navigate to the project directory:
sh
Code kopieren
cd mvsr-lab
Install dependencies:
sh
Code kopieren
pip install -r scripts/requirements.txt
Usage
To use this project, follow these steps:

Prepare the data:
Run the Jupyter notebook to prepare markers:

sh
Code kopieren
jupyter notebook scripts/prepare_markers.ipynb
Train the model:
Run the Jupyter notebook to train the model:

sh
Code kopieren
jupyter notebook scripts/train_model.ipynb
Detect markers:
Run the detection script:

sh
Code kopieren
python scripts/detect_markers.py
