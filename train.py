from dataset import BeeDataSet
import keras

dataset = BeeDataSet(source_dir='../dataset_genus8')
dataset.load()


