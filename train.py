from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from sklearn.utils import compute_class_weight
from runs import runs
from dataset import BeeDataSet
from model import DeepABIS
import keras
from keras import backend as K
import numpy as np
import os


def scheduler(epoch):
    learning_rate_init = 0.003
    if runs.current().lr_decay:
        if epoch > 40:
            learning_rate_init = 0.0005
        if epoch > 50:
            learning_rate_init = 0.0001
    return learning_rate_init


class LossWeightsModifier(keras.callbacks.Callback):
    def __init__(self, alpha, beta):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def on_epoch_end(self, epoch, logs={}):
        if epoch == 8:
            K.set_value(self.alpha, 0.9)
            K.set_value(self.beta, 0.1)
        if epoch == 18:
            K.set_value(self.alpha, 0.3)
            K.set_value(self.beta, 0.7)
        if epoch == 28:
            K.set_value(self.alpha, 0)
            K.set_value(self.beta, 1)


dataset = BeeDataSet(source_dir=runs.current().dataset, dataset_id=runs.current().id, input_shape=runs.current().input_shape)
dataset.load(mode=runs.current().mode)

class_weight = []
if runs.current().class_weights:
    class_weight_genus = compute_class_weight('balanced', np.unique(np.argmax(dataset.y_genus_train, axis=1)),
                                              np.argmax(dataset.y_genus_train, axis=1))
    class_weight_species = compute_class_weight('balanced', np.unique(np.argmax(dataset.y_species_train, axis=1)),
                                                np.argmax(dataset.y_species_train, axis=1))
    if runs.current().branches:
        class_weight = [class_weight_genus, class_weight_species]
    else:
        class_weight = [class_weight_species]

net = DeepABIS(dataset.num_genus, dataset.num_species, run=runs.current())
model = net.model()

# hyperparameters
batch_size = runs.current().batch_size
epochs = runs.current().epochs

# file paths
weights_store_filepath = './models/'
train_id = runs.current().id
log_filepath = './logs/' + train_id
model_name = 'deepabis_' + train_id + '.h5'
model_name_best = 'deepabis_' + train_id + '.weights.best.hdf5'
model_path = os.path.join(weights_store_filepath, model_name)
model_path_best = os.path.join(weights_store_filepath, model_name_best)

change_lr = LearningRateScheduler(scheduler)
tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0, batch_size=batch_size)
change_lw = LossWeightsModifier(net.alpha, net.beta)
checkpoint = ModelCheckpoint(model_path_best, monitor=('val_predictions_acc' if runs.current().branches else 'val_acc'),
                             verbose=1, save_best_only=True, mode='max')

if runs.current().branches:
    cbks = [change_lr, tb_cb, change_lw]
    model.fit_generator(
        dataset.Generator(dataset.train['path'].values, [dataset.y_genus_train, dataset.y_species_train],
                          dataset.scaler, input_shape=dataset.input_shape, batch_size=batch_size),
        class_weight=class_weight,
        steps_per_epoch=len(dataset.y_species_train) / batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=cbks,
        validation_data=(dataset.x_test, [dataset.y_genus_test, dataset.y_species_test]))
else:
    cbks = [change_lr, tb_cb, checkpoint]
    model.fit_generator(
        dataset.Generator(dataset.train['path'].values, dataset.y_species_train, dataset.scaler, dataset.input_shape, batch_size=batch_size),
        class_weight=class_weight,
        steps_per_epoch=len(dataset.y_species_train) / batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=cbks,
        validation_data=(dataset.x_test, dataset.y_species_test))

model.compile(loss='categorical_crossentropy',
              # optimizer=keras.optimizers.Adadelta(),
              optimizer=model.optimizer,
              metrics=['accuracy'])

if runs.current().branches:
    score = model.evaluate(dataset.x_test, [dataset.y_genus_test, dataset.y_species_test], verbose=0)
else:
    score = model.evaluate(dataset.x_test, dataset.y_species_test, verbose=0)
print('score is: ', score)

model.save(model_path)
