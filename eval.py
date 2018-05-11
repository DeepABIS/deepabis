from keras import Model
from keras.models import load_model
from sklearn.metrics import classification_report, precision_recall_fscore_support
from dataset import BeeDataSet
import pandas as pd
import numpy as np
import os

train_id = '2'


def pandas_classification_report(y_true, y_pred, target_names):
    labels = [i for i in range(len(target_names))]
    metrics_summary = precision_recall_fscore_support(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels)
    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
    class_report_df = pd.DataFrame(
        list(metrics_summary),
        index=metrics_sum_index,
        columns=target_names)
    total = class_report_df.loc[metrics_sum_index[-1]].sum()
    avg = (class_report_df.loc[metrics_sum_index[:-1]] * class_report_df.loc[metrics_sum_index[-1]]).sum(axis=1) / total
    class_report_df['avg / total'] = avg.tolist() + [total]
    return class_report_df.T


reports_filepath = './reports/'
weights_store_filepath = './models/'

model_name = 'beenet_' + train_id + '.h5'
model_path = os.path.join(weights_store_filepath, model_name)
model = load_model(model_path)

dataset = BeeDataSet(source_dir='../dataset_genus8')
dataset.load()

y_genus_test = np.argmax(dataset.y_genus_test, axis=1)

if train_id == '1':
    # Run 1 had one class too many (because 'embeddings.json' was interpreted as a class)
    y_genus_test_9 = np.zeros((dataset.y_genus_test.shape[0],9))
    y_genus_test_9[:,:-1] = dataset.y_genus_test
    y_genus_test = np.argmax(y_genus_test_9, axis=1)

y_genus_pred, y_species_pred = model.predict(dataset.x_test, verbose=1)
y_genus_pred = np.argmax(y_genus_pred, axis=1)
genus_report = pandas_classification_report(y_genus_test, y_genus_pred, dataset.genus_names)
genus_report.to_csv(reports_filepath + 'run_' + train_id + '_genus.csv')
print(genus_report)

y_species_test = np.argmax(dataset.y_species_test, axis=1)
y_species_pred = np.argmax(y_species_pred, axis=1)
species_report = pandas_classification_report(y_species_test, y_species_pred, dataset.species_names)
species_report.to_csv(reports_filepath + 'run_' + train_id + '_species.csv')
print(species_report)

