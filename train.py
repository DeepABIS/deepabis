from keras.callbacks import LearningRateScheduler, TensorBoard
from sklearn.utils import compute_class_weight
from runs import runs
from dataset import BeeDataSet
from model import BeeCNN
import keras
from keras import backend as K
import numpy as np
import os


def scheduler(epoch):
    learning_rate_init = 0.003
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


dataset = BeeDataSet(source_dir=runs.current().dataset)
dataset.load()

class_weight_genus = compute_class_weight('balanced', np.unique(np.argmax(dataset.y_genus_train, axis=1)),
                                                                np.argmax(dataset.y_genus_train, axis=1))
class_weight_species = compute_class_weight('balanced', np.unique(np.argmax(dataset.y_species_train, axis=1)),
                                                                  np.argmax(dataset.y_species_train, axis=1))

net = BeeCNN(dataset.num_genus, dataset.num_species)
model = net.model()

# hyperparameters
batch_size = runs.current().batch_size
epochs = runs.current().epochs

# file paths
weights_store_filepath = './models/'
train_id = runs.current().id
log_filepath = './logs/run' + train_id
model_name = 'beenet_' + train_id + '.h5'
model_path = os.path.join(weights_store_filepath, model_name)

change_lr = LearningRateScheduler(scheduler)
tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
change_lw = LossWeightsModifier(net.alpha, net.beta)
cbks = [change_lr, tb_cb, change_lw]

model.fit(dataset.x_train, [dataset.y_genus_train, dataset.y_species_train],
          class_weight=[class_weight_genus, class_weight_species],
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=cbks,
          validation_data=(dataset.x_test, [dataset.y_genus_test, dataset.y_species_test]))

# ---------------------------------------------------------------------------------
# The following compile() is just a behavior to make sure this model can be saved.
# We thought it may be a bug of Keras which cannot save a model compiled with loss_weights parameter
# ---------------------------------------------------------------------------------
model.compile(loss='categorical_crossentropy',
              # optimizer=keras.optimizers.Adadelta(),
              optimizer=model.optimizer,
              metrics=['accuracy'])

score = model.evaluate(dataset.x_test, [dataset.y_genus_test, dataset.y_species_test], verbose=0)
model.save(model_path)
print('score is: ', score)
