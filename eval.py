import itertools

import keras
from tqdm import trange
from keras.utils import plot_model, CustomObjectScope
from matplotlib import pyplot as plt
from keras.models import load_model
from keras import backend as K
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix, accuracy_score
from sklearn import svm
from dataset import BeeDataSet
from runs import runs
from helpers import top_k_accuracy_score
import pandas as pd
import numpy as np
import os

train_id = runs.current().id


def predict_with_svm(train, test, batch_size = 16):
    feature_layer = model.get_layer('fc2')
    get_activations = K.function([model.layers[0].input], [feature_layer.output])
    # Train
    features = np.zeros((train.shape[0], feature_layer.output.shape[1]))
    for i in trange(int(np.ceil(train.shape[0] / batch_size))):
        batch = train[i * batch_size:(i + 1) * batch_size]
        feature = get_activations([batch])[0]
        features[i * batch_size:(i + 1) * batch_size] = feature
    clf = svm.SVC(kernel='poly', class_weight='balanced')
    clf.fit(features, np.argmax(dataset.y_species_train, axis=1))
    # Test
    test_features = np.zeros((test.shape[0], feature_layer.output.shape[1]))
    for i in trange(int(np.ceil(test.shape[0] / batch_size))):
        batch = test[i * batch_size:(i + 1) * batch_size]
        feature = get_activations([batch])[0]
        test_features[i * batch_size:(i + 1) * batch_size] = feature
    return clf.predict(test_features)


def pandas_classification_report(y_true, y_pred, target_names, weighted=True):
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
    total = class_report_df.loc[metrics_sum_index[-1]].sum() if weighted else len(target_names)
    weights = class_report_df.loc[metrics_sum_index[-1]] if weighted else 1
    avg = (class_report_df.loc[metrics_sum_index[:-1]] * weights).sum(axis=1) / total
    class_report_df['avg / total'] = avg.tolist() + [total]
    return class_report_df.T


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

    assert cm.shape[0] == len(classes)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
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


reports_filepath = './reports/' + train_id + '/'
if not os.path.exists(reports_filepath):
    os.mkdir(reports_filepath)
weights_store_filepath = './models/'

model_name = 'deepabis_' + train_id + ('.weights.best.hdf5' if not runs.current().branches else '.h5')
model_path = os.path.join(weights_store_filepath, model_name)
if runs.current().model in ('mobilenet', 'mobilenetV2', 'mobilenetV2_pretrained'):
    def relu6(x):
        return keras.backend.relu(x, max_value=6)
    with CustomObjectScope({'relu6': relu6}):
        model = load_model(model_path)
else:
    model = load_model(model_path)
plot_model(model, to_file=reports_filepath + '/model.png', show_shapes=False, show_layer_names=False, rankdir='TB')

dataset = BeeDataSet(source_dir=runs.current().dataset, dataset_id=runs.current().id, input_shape=runs.current().input_shape)
dataset.load(mode=runs.current().mode, test_only=True)

y_genus_test = np.argmax(dataset.y_genus_test, axis=1)

if runs.current().branches:
    y_genus_pred, y_species_pred = model.predict(dataset.x_test, verbose=1)
    y_genus_pred = np.argmax(y_genus_pred, axis=1)
    genus_report = pandas_classification_report(y_genus_test, y_genus_pred, dataset.genus_names)
    genus_report.to_csv(reports_filepath + '/genus.csv')

    cnf_matrix_genus = confusion_matrix(y_genus_test, y_genus_pred, labels=dataset.genus_index)

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix_genus, classes=dataset.genus_names,
                          title='Genus confusion matrix, with normalization', normalize=True)
    plt.savefig(reports_filepath + '/genus.png')
    print('Top-1: {}'.format(accuracy_score(np.argmax(dataset.y_species_test, axis=1), np.argmax(y_species_pred, axis=1))))
    print('Top-5: {}'.format(top_k_accuracy_score(np.argmax(dataset.y_species_test, axis=1), y_species_pred)))
else:
    print(model.count_params())
    y_species_pred = model.predict(dataset.x_test, verbose=1)
    print('Top-1: {}'.format(accuracy_score(np.argmax(dataset.y_species_test, axis=1), np.argmax(y_species_pred, axis=1))))
    print('Top-5: {}'.format(top_k_accuracy_score(np.argmax(dataset.y_species_test, axis=1), y_species_pred)))

y_species_test = np.argmax(dataset.y_species_test, axis=1)
y_species_pred = np.argmax(y_species_pred, axis=1)
for weighted in [True, False]:
    species_report = pandas_classification_report(y_species_test, y_species_pred, dataset.species_names, weighted=weighted)
    species_report.to_csv('{}/species{}.csv'.format(reports_filepath, '_regular' if not weighted else ''))

np.set_printoptions(precision=2)

# Compute confusion matrix
cnf_matrix_species = confusion_matrix(y_species_test, y_species_pred, labels=dataset.species_index)
# Plot normalized confusion matrix
plt.figure(figsize=(20, 20))
plot_confusion_matrix(cnf_matrix_species, classes=dataset.species_names,
                      title='Species confusion matrix, with normalization', normalize=True, plot_text=False)
plt.savefig(reports_filepath + '/species.png')