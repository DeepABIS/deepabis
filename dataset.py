import pandas as pd
import json
import glob
import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import keras


class BeeDataSet:
    def __init__(self, source_dir):
        self.source_dir = source_dir
        self.genera = {}
        self.embedding = {}
        self.num_genus = 0
        self.num_species = 0
        self.num_files = 0
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

    def load(self):
        num_species = 0
        num_files = 0
        genera = {}
        for genus_path in glob.iglob(self.source_dir + '/*'):
            if not os.path.isdir(genus_path):
                continue
            genus = os.path.basename(genus_path)
            genera[genus] = {}
            for species_path in glob.iglob(genus_path + '/*'):
                species = os.path.basename(species_path)
                genera[genus][species] = []
                num_species += 1
                for sex_path in glob.iglob(species_path + '/*'):
                    sex = os.path.basename(sex_path)
                    for filename in glob.iglob(sex_path + '/*.JPG'):
                        basename = os.path.basename(filename)
                        genera[genus][species].append(filename)
                        num_files += 1
        self.genera = genera
        self.num_genus = len(self.genera.keys())
        self.num_species = num_species
        self.num_files = num_files
        with open(self.source_dir + '/embeddings.json', 'rb') as file:
            self.embedding = json.load(file)
        self.make_data()

    def make_data(self):
        x = []
        y_genus = []
        y_species = []
        with tqdm(total=self.num_files) as pbar:
            for genus in self.genera:
                for species in self.genera[genus]:
                    for filename in self.genera[genus][species]:
                        with open(filename, 'rb') as stream:
                            bytes = bytearray(stream.read())
                            numpyarray = np.asarray(bytes, dtype=np.uint8)
                            img = cv2.imdecode(numpyarray, cv2.IMREAD_GRAYSCALE)
                            img = cv2.resize(img, (256, 256))
                            img = np.float32(img)
                            img = np.reshape(img, (256, 256, 1))
                            label_genus, label_species = self.get_embedding(genus, species)
                            x.append(img)
                            y_genus.append(label_genus)
                            y_species.append(label_species)
                            pbar.update()
        x = np.array(x)
        y_genus = np.array(y_genus)
        y_species = np.array(y_species)
        self.x_train, self.x_test, self.y_genus_train, self.y_genus_test, self.y_species_train, self.y_species_test = \
            train_test_split(x, y_genus, y_species, test_size = 0.10, random_state = 42)

        self.x_train = (self.x_train - np.mean(self.x_train)) / np.std(self.x_train)
        self.x_test = (self.x_test - np.mean(self.x_test)) / np.std(self.x_test)

        self.y_genus_train = keras.utils.to_categorical(self.y_genus_train, self.num_genus)
        self.y_genus_test = keras.utils.to_categorical(self.y_genus_test, self.num_genus)

        self.y_species_train = keras.utils.to_categorical(self.y_species_train, self.num_species)
        self.y_species_test = keras.utils.to_categorical(self.y_species_test, self.num_species)