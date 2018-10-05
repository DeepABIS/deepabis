import math

from keras.utils import Sequence
from matplotlib import pyplot as plt
import pandas as pd
import json
import glob
import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import keras
from sklearn.preprocessing import StandardScaler
import joblib
from dask import array as da

tqdm.pandas()


class BeeDataSet:
    def __init__(self, source_dir, dataset_id):
        self.source_dir = source_dir
        self.dataset_id = dataset_id
        self.df = None
        self.genus_names = []
        self.species_names = []
        self.embedding = {}
        self.num_genus = 0
        self.num_species = 0
        self.num_files = 0
        self.train = None
        self.test = None
        self.x_train = np.array([])
        self.y_genus_train = np.array([])
        self.x_test = np.array([])
        self.y_genus_test = np.array([])
        self.y_species_train = np.array([])
        self.y_species_test = np.array([])
        self.scaler = StandardScaler()

    def get_embedding(self, genus, species=None):
        genus_embedding = self.embedding[genus]['index']
        species_embedding = None
        if species is not None:
            species_embedding = self.embedding[genus]['species'][species]
        return genus_embedding, species_embedding

    def load(self, mode = 'mean_subtraction', test_only = False):
        mode_options = ('mean_subtraction', 'per_channel')
        if mode not in mode_options:
            raise ValueError('Mode has to be one of ' + str(mode_options))
        # Load embedding (genus/species --> index)
        with open(self.source_dir + '/embeddings.json', 'rb') as file:
            self.embedding = json.load(file)
        genus_names = []
        genus_indices = []
        species_names = []
        species_indices = []
        for genus in self.embedding:
            index = self.embedding[genus]['index']
            genus_indices.append(index)
            genus_names.append(genus)
            for species in self.embedding[genus]['species']:
                species_index = self.embedding[genus]['species'][species]
                species_indices.append(species_index)
                species_names.append(genus + ' ' + species)
        genus_df = pd.DataFrame(data={'index':genus_indices}, index=genus_names)
        species_df = pd.DataFrame(data={'index':species_indices}, index=species_names)

        # Load number of files
        num_files = 0
        num_train = 0
        num_test = 0
        for type_path in glob.iglob(self.source_dir + '/*'):
            type = os.path.basename(type_path)
            if test_only and type == 'train':
                continue
            if not os.path.isdir(type_path):
                continue
            for genus_path in glob.iglob(type_path + '/*'):
                genus = os.path.basename(genus_path)
                genus_index = self.get_embedding(genus)
                for species_path in glob.iglob(genus_path + '/*'):
                    species = os.path.basename(species_path)
                    species_index = self.get_embedding(genus, species)
                    for filename in glob.iglob(species_path + '/*.JPG'):
                        num_files += 1
                        if type == 'train':
                            num_train += 1
                        else:
                            num_test += 1
        paths = []
        x = []
        y_genera = []
        y_species = []
        types = []

        self.x_train = np.zeros((num_train, 224, 224, 3))
        self.x_test = np.zeros((num_test, 224, 224, 3))
        test_i = 0

        if not test_only:
            self.scaler = StandardScaler(with_std=mode == 'per_channel')

        # Load images
        with tqdm(total=num_files) as pbar:
            for type in ['train', 'test']:
                if test_only and type == 'train':
                    continue
                for genus_path in glob.iglob(self.source_dir + '/' + type + '/*'):
                    genus = os.path.basename(genus_path)
                    for species_path in glob.iglob(genus_path + '/*'):
                        species = os.path.basename(species_path)
                        genus_index, species_index = self.get_embedding(genus, species)
                        for filename in glob.iglob(species_path + '/*.JPG'):
                            basename = os.path.basename(filename)
                            paths.append(filename)
                            y_genera.append(genus_index)
                            y_species.append(species_index)
                            types.append(type)
                            with open(filename, 'rb') as stream:
                                bytes = bytearray(stream.read())
                                numpyarray = np.asarray(bytes, dtype=np.uint8)
                                img = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
                                img = cv2.resize(img, (224, 224))
                                img = np.float32(img)
                                img = np.reshape(img, (224, 224, 3))
                                if type == 'train':
                                    self.scaler.partial_fit(img.reshape(-1, 1))
                                if type == 'test':
                                    self.x_test[test_i] = img
                                    test_i += 1
                            pbar.update()

        scaler_path = './transform/' + str(self.dataset_id) + '.pkl'
        if not test_only:
            print('Mean: {}'.format(self.scaler.mean_))
            print('Std: {}'.format(self.scaler.scale_))
            joblib.dump(self.scaler, scaler_path)
        else:
            self.scaler = joblib.load(scaler_path)

        data = {
            'path': paths,
            'set': pd.Categorical(types),
            'genus': pd.Categorical(np.array(y_genera)),
            'species': pd.Categorical(np.array(y_species))
        }
        print('Making dataframe...')
        self.df = pd.DataFrame(data=data)
        self.num_genus = genus_df.shape[0]
        self.num_species = species_df.shape[0]
        self.num_files = num_files
        self.genus_names = genus_df.index.values
        self.species_names = species_df.index.values
        self.species_index = species_df['index']
        print('Extracting train data...')
        self.train = self.df[self.df.set == 'train']
        print('Extracting test data...')
        self.test = self.df[self.df.set == 'test']
        self.y_genus_train = self.train['genus'].values
        self.y_genus_test = self.test['genus'].values
        self.y_species_train = self.train['species'].values
        self.y_species_test = self.test['species'].values
        print('Transforming data...')
        self.transform_data(mode=mode)

    def transform_data(self, mode='mean_subtraction'):
        print('Normalizing data (' + mode + ')...')
        self.x_test = self.transform(self.x_test)
        if mode != 'per_channel':
            self.x_test /= 255

        print('To Categorical...')
        self.y_genus_train = keras.utils.to_categorical(self.y_genus_train, self.num_genus)
        self.y_genus_test = keras.utils.to_categorical(self.y_genus_test, self.num_genus)

        self.y_species_train = keras.utils.to_categorical(self.y_species_train, self.num_species)
        self.y_species_test = keras.utils.to_categorical(self.y_species_test, self.num_species)

    def transform(self, X):
        X -= self.scaler.mean_
        X /= self.scaler.scale_
        return X

    class Generator(Sequence):
        # Class is a dataset wrapper for better training performance
        def __init__(self, x_set, y_set, scaler, batch_size=32):
            self.x, self.y = x_set, y_set
            self.scaler = scaler
            self.batch_size = batch_size
            self.indices = np.arange(self.x.shape[0])
            #np.random.shuffle(self.indices)

        def __len__(self):
            return math.ceil(self.x.shape[0] / self.batch_size)

        def __getitem__(self, idx):
            inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_x = np.zeros((len(inds), 224, 224, 3))
            i = 0
            for index in inds:
                path = self.x[index]
                with open(path, 'rb') as stream:
                    bytes = bytearray(stream.read())
                    numpyarray = np.asarray(bytes, dtype=np.uint8)
                    img = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
                    img = cv2.resize(img, (224, 224))
                    img = np.float32(img)
                    img = np.reshape(img, (224, 224, 3))
                    batch_x[i] = img
                    i+=1

            batch_x = self.transform(batch_x)
            if isinstance(self.y, list):
                batch_y = {'c1_predictions': self.y[0][inds], 'predictions': self.y[1][inds]}
            else:
                batch_y = self.y[inds]
            return batch_x, batch_y

        def on_epoch_end(self):
            np.random.shuffle(self.indices)

        def transform(self, X):
            X -= self.scaler.mean_
            X /= self.scaler.scale_
            return X