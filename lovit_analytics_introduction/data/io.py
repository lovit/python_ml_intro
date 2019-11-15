import os
import pandas as pd


data_dir = f'{os.path.dirname(__file__)}/datafile/'

pima_indians_data_paths = [
    f'{data_dir}pima-indians-diabetes.data',
    f'{data_dir}pima-indians-diabetes.names',
]

def readtxt(path):
    with open(path, encoding='utf-8') as f:
        lines = [line.strip() for line in f]
    return lines

def load_pima_indians_data():
    names = readtxt(pima_indians_data_paths[1])
    data = pd.read_csv(pima_indians_data_paths[0], names=names)
    return data
