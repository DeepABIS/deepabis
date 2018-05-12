import itertools

from matplotlib import pyplot as plt
from keras import Model
from keras.models import load_model
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix
from dataset import BeeDataSet
import pandas as pd
import numpy as np
import os

train_id = '3'


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

dataset = BeeDataSet(source_dir='../dataset_genus8_traintest')
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


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          plot_text=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    if plot_text:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Compute confusion matrix
cnf_matrix_genus = confusion_matrix(y_genus_test, y_genus_pred)
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix_genus, classes=dataset.genus_names,
                      title='Genus confusion matrix, with normalization', normalize=True)
plt.savefig('reports/run_' + train_id + '_genus.png')

# Compute confusion matrix
cnf_matrix_species = confusion_matrix(y_species_test, y_species_pred)

# Plot normalized confusion matrix
plt.figure(figsize=(20, 20))
plot_confusion_matrix(cnf_matrix_species, classes=dataset.species_names,
                      title='Species confusion matrix, with normalization', normalize=True, plot_text=False)
plt.savefig('reports/run_' + train_id + '_species.png')

