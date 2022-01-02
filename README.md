# Advanced Machine Learning Project

In this project, the aim was to retrieve the results of a new method based on deep neural network describe into an research article. In this package we try to reproduce the Hierarchical Self-Attention Network for Action Localization in Videos (HISAN) method of Rizard Renanda Adhi Pramono, Yie-Tarng Chen, and Wen-Hsien Fang published in ICCV in 2019.

## Prerequisites

- python >= 3.7

## Dependencies

- numpy, tqdm, pandas, tensorflow, keras
- pathlib, pickle, shutil, jupyter

## Development

Please download the current package to your local directory. To use this package you need to download all dependencies. You can either use the setup file or the requirements file.

You will have to go to this directory path where setup.py is located and use the command line below to install the dependencies needed.

```bash
setup.py install
```

Or do the same with this commande file to use the requirements file.

```bash
pip install -r requirements.txt
```

## Usage

To use the code you need to use open the folder as your working project on your IDE. To use Colab, it is necessary to give authority to Colab for connecting your Google Drive, upload the folder to your Drive and importing the path in your notebook.

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Process data

In this package you will find the JHMDB and UCF101-24 datasets containing frames, optical flow frames and their ground truth. We obtain these data from MMAction2 Contributors [Lien](https://github.com/open-mmlab/mmaction2 "OpenMMLab's Next Generation Video Understanding Toolbox and Benchmark").
To process these data to use the algorithm, it is necessary to lunch the file "process_data.py" that will sort those data and split for Train and Test process. Also it generate csv file that contain all the sorted informations needed to continue.

So first download at their google drive [Lien](https://drive.google.com/drive/folders/1BvGywlAGrACEqRyfYbz3wzlVV3cDFkct) the following files : UCF101_v2.tar.gz and JHMDB.tar.gz and place them into a folder name "datasets/" in the main folder "ProjetMLA/" that contains the package.
Type these commande to uncompress.

```bash
tar -zxvf UCF101_v2.tar.gz
tar -zxvf JHMDB.tar.gz
```

Go to the project folder path on your terminal and lunch the file (it can take some times) :

```bash
python3 process_data.py
```

### Training of the models

#### Faster RCNN

First it is needed to train the Faster RCNN on his own with the file Training_fasterRCNN.py.

#### HISAN

Trainning of the Hisan : HISAN_train.py
