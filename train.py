from keras.callbacks import LearningRateScheduler

from dataset import BeeDataSet
from model import BeeCNN
import keras
from keras import backend as K
import cv2
import numpy as np


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
      K.set_value(self.alpha, 0.1)
      K.set_value(self.beta, 0.8)
    if epoch == 18:
      K.set_value(self.alpha, 0.1)
      K.set_value(self.beta, 0.2)
    if epoch == 28:
      K.set_value(self.alpha, 0)
      K.set_value(self.beta, 0)


dataset = BeeDataSet(source_dir='../dataset_genus8')
dataset.load()

net = BeeCNN(dataset.num_genus, dataset.num_species)
model = net.model()

batch_size = 32
epochs = 60

change_lr = LearningRateScheduler(scheduler)
change_lw = LossWeightsModifier(net.alpha, net.beta)
cbks = [change_lr, change_lw]

model.fit(dataset.x_train, [dataset.y_genus_train, dataset.y_species_train],
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=cbks,
          validation_data=(dataset.x_test, [dataset.y_genus_test, dataset.y_species_test]))
