import configparser
from collections import namedtuple

Run = namedtuple('Run', 'id dataset batch_size epochs mode class_weights lr_decay model branches optimizer input_shape')


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
                mode = run['mode']
                class_weights = run.getboolean('class_weights')
                lr_decay = run.getboolean('lr_decay')
                model = run['model']
                branches = run.getboolean('branches')
                optimizer = run['optimizer']
                input_shape = tuple(map(int, run['input_shape'].split(',')))
                self.runs[section] = Run(id=id,
                                         dataset=dataset,
                                         batch_size=batch_size,
                                         epochs=epochs,
                                         mode=mode,
                                         class_weights=class_weights,
                                         lr_decay=lr_decay,
                                         model=model,
                                         branches=branches,
                                         optimizer=optimizer,
                                         input_shape=input_shape)
        if 'CURRENT' in config:
            self.run = config['CURRENT']['run']

    def current(self):
        return self.runs[self.run]

    def __getitem__(self, item):
        return self.runs[item]


runs = Runs()