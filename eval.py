from keras import Model
from dataset import BeeDataSet
from model import BeeCNN
import os

weights_store_filepath = './models/'
train_id = '1'
model_name = 'beenet_' + train_id + '.h5'
model_path = os.path.join(weights_store_filepath, model_name)

dataset = BeeDataSet(source_dir='../dataset_genus8')
dataset.load()

net = BeeCNN(dataset.num_genus, dataset.num_species)
model = net.model()
model.load_weights(model_path)
score = model.evaluate(dataset.x_test, [dataset.y_genus_test, dataset.y_species_test], verbose=1)
print(model.metrics_names)
print(score)


