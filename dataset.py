import pandas as pd
import json
import glob
import os


class BeeDataSet:
    def __init__(self, source_dir):
        self.source_dir = source_dir
        self.genera = {}
        self.embedding = {}

    def get_embedding(self, genus, species=None):
        genus_embedding = self.embedding[genus]['index']
        species_embedding = None
        if species is not None:
            species_embedding = self.embedding[genus]['species'][species]
        return genus_embedding, species_embedding

    def load(self):
        genera = {}
        for genus_path in glob.iglob(self.source_dir + '/*'):
            genus = os.path.basename(genus_path)
            genera[genus] = {}
            for species_path in glob.iglob(genus_path + '/*'):
                species = os.path.basename(species_path)
                genera[genus][species] = []
                for sex_path in glob.iglob(species_path + '/*'):
                    sex = os.path.basename(sex_path)
                    for filename in glob.iglob(sex_path + '/*.JPG'):
                        basename = os.path.basename(filename)
                        genera[genus][species].append(filename)
        self.genera = genera
        with open(self.source_dir + '/embeddings.json', 'rb') as file:
            self.embedding = json.load(file)

