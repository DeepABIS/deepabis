import configparser
from collections import namedtuple

Run = namedtuple('Run', 'id dataset batch_size epochs')


class Runs:
    def __init__(self):
        config = configparser.ConfigParser()
        config.read('runs.ini')
        self.runs = {}
        self.run = None
        for section in config.sections():
            if section != 'CURRENT':
                if self.run is None:
                    self.run = section
                run = config[section]
                id = run['id']
                dataset = run['dataset']
                batch_size = int(run['batch_size'])
                epochs = int(run['epochs'])
                self.runs[section] = Run(id=id, dataset=dataset, batch_size=batch_size, epochs=epochs)
        if 'CURRENT' in config:
            self.run = config['CURRENT']['run']

    def current(self):
        return self.runs[self.run]

    def __getitem__(self, item):
        return self.runs[item]


runs = Runs()