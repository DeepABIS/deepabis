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
import dask.array as da

tqdm.pandas()


class BeeDataSet:
    def __init__(self, source_dir):
        self.source_dir = source_dir
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

    def get_embedding(self, genus, species=None):
        genus_embedding = self.embedding[genus]['index']
        species_embedding = None
        if species is not None:
            species_embedding = self.embedding[genus]['species'][species]
        return genus_embedding, species_embedding

    def load(self, mode = 'mean_subtraction', test_only = False):
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
                species_names.append(species)
        genus_df = pd.DataFrame(data={'index':genus_indices}, index=genus_names)
        species_df = pd.DataFrame(data={'index':species_indices}, index=species_names)

        # Load number of files
        num_files = 0
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
        paths = []
        x = []
        y_genera = []
        y_species = []
        types = []
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
                            with open(filename, 'rb') as stream:
                                bytes = bytearray(stream.read())
                                numpyarray = np.asarray(bytes, dtype=np.uint8)
                                img = cv2.imdecode(numpyarray, cv2.IMREAD_GRAYSCALE)
                                img = cv2.resize(img, (256, 256))
                                img = np.float32(img)
                                img = np.reshape(img, (256, 256, 1))
                                x.append(img)
                                paths.append(filename)
                                y_genera.append(genus_index)
                                y_species.append(species_index)
                                types.append(type)
                                pbar.update()
        data = {
            'img': x,
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
        print('Extracting train data...')
        self.train = self.df[self.df.set == 'train']
        print('Extracting test data...')
        self.test = self.df[self.df.set == 'test']
        self.x_train = np.array([x[index] for index, row in self.train.iterrows()])
        self.x_test = np.array([x[index] for index, row in self.test.iterrows()])
        self.y_genus_train = self.train['genus'].values
        self.y_genus_test = self.test['genus'].values
        self.y_species_train = self.train['species'].values
        self.y_species_test = self.test['species'].values
        print('Transforming data...')
        self.transform_data(mode=mode)

    def transform_data(self, mode = 'mean_subtraction'):
        mode_options = ('mean_subtraction', 'per_channel')
        if mode not in mode_options:
            raise ValueError('Mode has to be one of ' + str(mode_options))
        print('Normalizing data (' + mode + ')...')
        if self.x_train.shape[0] > 0:
            train_da = da.from_array(self.x_train, chunks=(500, 256, 256, 1))
            train_mean = train_da.mean().compute()
            self.x_train = (self.x_train - train_mean)
            if mode == 'per_channel':
                std = train_da.std().compute()
                self.x_train /= std
            else:
                self.x_train /= 255

        test_da = da.from_array(self.x_test, chunks=(500, 256, 256, 1))
        test_mean = test_da.mean().compute()
        self.x_test = (self.x_test - test_mean)
        if mode == 'per_channel':
            std = test_da.std().compute()
            self.x_test /= std
        else:
            self.x_test /= 255

        print('To Categorical...')
        self.y_genus_train = keras.utils.to_categorical(self.y_genus_train, self.num_genus)
        self.y_genus_test = keras.utils.to_categorical(self.y_genus_test, self.num_genus)

        self.y_species_train = keras.utils.to_categorical(self.y_species_train, self.num_species)
        self.y_species_test = keras.utils.to_categorical(self.y_species_test, self.num_species)