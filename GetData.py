import csv
from pathlib import Path
import pandas as pd
import os

filePath = __file__  # This script file path
parentFolder = str(Path(filePath).parent)   # Folder of this file --> "Project"
csv_filename="Data//RGB//UCF101//data_RGB_train.csv"

train_data=pd.read_csv(parentFolder+"//"+csv_filename , sep=";")
filePath = os.path.abspath('')
print(filePath)
print(train_data.head())
print(train_data["file_name"].tolist()[0])
train_data=train_data.values.tolist()
print(train_data[0])