from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Model
from keras import backend as K
from keras import optimizers


class BeeCNN:
    def __init__(self, num_genus, num_species, version='baseline'):
        img_rows, img_cols = 256, 256
        if K.image_data_format() == 'channels_first':
            self.input_shape = (1, img_rows, img_cols)
        else:
            self.input_shape = (img_rows, img_cols, 1)
        self.num_c_1 = num_genus
        self.num_classes = num_species

        self.alpha = K.variable(value=0.99, dtype="float32", name="alpha")  # A1 in paper
        self.beta = K.variable(value=0.01, dtype="float32", name="beta")  # A2 in paper

    def model(self):
        img_input = Input(shape=self.input_shape, name='input')

        # --- block 1 ---
        x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
        x = BatchNormalization()(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # --- block 2 ---
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # --- coarse 1 branch ---
        c_1_bch = Flatten(name='c1_flatten')(x)
        c_1_bch = Dense(64, activation='relu', name='c1_fc_1')(c_1_bch)
        c_1_bch = BatchNormalization()(c_1_bch)
        c_1_bch = Dropout(0.5)(c_1_bch)
        c_1_pred = Dense(self.num_c_1, activation='softmax', name='c1_predictions')(c_1_bch)

        # --- block 3 ---
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # --- fine block ---
        x = Flatten(name='flatten')(x)
        x = Dense(256, activation='relu', name='fc_1')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu', name='fc2')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        fine_pred = Dense(self.num_classes, activation='softmax', name='predictions')(x)

        model = Model(inputs=img_input, outputs=[c_1_pred, fine_pred], name='beecnn')
        model.summary()

        sgd = optimizers.SGD(lr=0.003, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      loss_weights=[self.alpha, self.beta],
                      # optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy', 'top_k_categorical_accuracy'])
        return model
