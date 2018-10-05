from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout, \
    GlobalAveragePooling2D, Concatenate
from keras.models import Model
from keras import backend as K
from keras import optimizers
import keras


class BeeCNN:
    def __init__(self, num_genus, num_species, run):
        self.run = run
        img_rows, img_cols = 224, 224
        if K.image_data_format() == 'channels_first':
            self.input_shape = (1, img_rows, img_cols)
        else:
            self.input_shape = (img_rows, img_cols, 3)
        self.num_c_1 = num_genus
        self.num_classes = num_species
        self.version = run.model
        valid_optimizers = ('SGD', 'AdaDelta', 'AdaGrad', 'Adam')
        if run.optimizer not in valid_optimizers:
            raise ValueError('Optimizer has to be one of ' + str(valid_optimizers))
        self.alpha = K.variable(value=0.99, dtype="float32", name="alpha")
        self.beta = K.variable(value=0.01, dtype="float32", name="beta")

    def model(self):
        model_func = getattr(self, self.version)
        return model_func()

    def optimizer(self):
        if self.run.optimizer == 'SGD':
            return optimizers.SGD(lr=0.003, momentum=0.9, nesterov=True)
        if self.run.optimizer == 'AdaDelta':
            return optimizers.Adadelta()
        if self.run.optimizer == 'AdaGrad':
            return optimizers.Adagrad()
        if self.run.optimizer == 'Adam':
            return optimizers.Adam()

    def inception_resnet(self):
        model = keras.applications.InceptionResNetV2(input_shape=self.input_shape, weights=None, classes=self.num_classes)
        model.summary()

        model.compile(loss='categorical_crossentropy',
                      optimizer=self.optimizer(),
                      # optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy', 'top_k_categorical_accuracy'])
        return model

    def inception_resnet_pretrained(self):
        model = keras.applications.InceptionResNetV2(input_shape=self.input_shape,
                                                     include_top=False,
                                                     weights='imagenet',
                                                     classes=self.num_classes)

        x = GlobalAveragePooling2D(name='avg_pool')(model.output)
        x = Dense(self.num_classes, activation='softmax', name='predictions')(x)

        model = Model(model.input, x)

        model.summary()

        model.compile(loss='categorical_crossentropy',
                      optimizer=self.optimizer(),
                      metrics=['accuracy', 'top_k_categorical_accuracy'])
        return model

    def mobilenet(self):
        model = keras.applications.MobileNetV2(input_shape=self.input_shape, weights=None, classes=self.num_classes)
        model.summary()

        model.compile(loss='categorical_crossentropy',
                      optimizer=self.optimizer(),
                      # optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy', 'top_k_categorical_accuracy'])
        return model

    def mobilenetV2(self):
        model = keras.applications.MobileNetV2(input_shape=self.input_shape, weights=None, classes=self.num_classes)
        model.summary()

        model.compile(loss='categorical_crossentropy',
                      optimizer=self.optimizer(),
                      # optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy', 'top_k_categorical_accuracy'])
        return model

    def mobilenetV2_pretrained(self):
        model = keras.applications.MobileNetV2(input_shape=self.input_shape, include_top=False)
        last = model.output

        x = GlobalAveragePooling2D()(last)
        x = Dense(self.num_classes, activation='softmax',
                  use_bias=True, name='Logits')(x)

        model = Model(model.input, x)

        model.compile(loss='categorical_crossentropy',
                      optimizer=self.optimizer(),
                      # optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy', 'top_k_categorical_accuracy'])
        return model

    def oneloss(self):
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

        # --- block 3 ---
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # --- block 4 ---
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # --- fine block ---
        x = Flatten(name='flatten')(x)
        x = Dense(2048, activation='relu', name='fc_1')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(2048, activation='relu', name='fc2')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        fine_pred = Dense(self.num_classes, activation='softmax', name='predictions')(x)

        model = Model(inputs=img_input, outputs=[fine_pred], name='beecnn')
        model.summary()

        model.compile(loss='categorical_crossentropy',
                      optimizer=self.optimizer(),
                      # optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy', 'top_k_categorical_accuracy'])
        return model

    def blocks4(self):
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

        # --- block 4 ---
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # --- fine block ---
        x = Flatten(name='flatten')(x)
        x = Dense(512, activation='relu', name='fc_1')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation='relu', name='fc2')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        fine_pred = Dense(self.num_classes, activation='softmax', name='predictions')(x)

        model = Model(inputs=img_input, outputs=[c_1_pred, fine_pred], name='beecnn')
        model.summary()

        model.compile(loss='categorical_crossentropy',
                      optimizer=self.optimizer(),
                      loss_weights=[self.alpha, self.beta],
                      # optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy', 'top_k_categorical_accuracy'])
        return model

    def baseline(self):
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

        model.compile(loss='categorical_crossentropy',
                      optimizer=self.optimizer(),
                      loss_weights=[self.alpha, self.beta],
                      # optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy', 'top_k_categorical_accuracy'])
        return model

    def baseline2(self):
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

        # --- block 3 ---
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # --- coarse 1 branch ---
        c_1_bch = Flatten(name='c1_flatten')(x)
        c_1_bch = Dense(256, activation='relu', name='c1_fc_1')(c_1_bch)
        c_1_bch = BatchNormalization()(c_1_bch)
        c_1_bch = Dropout(0.5)(c_1_bch)
        c_1_pred = Dense(self.num_c_1, activation='softmax', name='c1_predictions')(c_1_bch)

        # --- block 4 ---
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # --- block 5 ---
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        # --- fine block ---
        x = Flatten(name='flatten')(x)
        x = Dense(1024, activation='relu', name='fc_1')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation='relu', name='fc2')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        fine_pred = Dense(self.num_classes, activation='softmax', name='predictions')(x)

        model = Model(inputs=img_input, outputs=[c_1_pred, fine_pred], name='beecnn')
        model.summary()

        model.compile(loss='categorical_crossentropy',
                      optimizer=self.optimizer(),
                      loss_weights=[self.alpha, self.beta],
                      # optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy', 'top_k_categorical_accuracy'])
        return model

    def baseline2_connected(self):
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

        # --- block 3 ---
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # --- coarse 1 branch ---
        c_1_bch = Flatten(name='c1_flatten')(x)
        c_1_bch = Dense(256, activation='relu', name='c1_fc_1')(c_1_bch)
        c_1_bch = BatchNormalization()(c_1_bch)
        c_1_bch = Dropout(0.5)(c_1_bch)
        c_1_pred = Dense(self.num_c_1, activation='softmax', name='c1_predictions')(c_1_bch)

        # --- block 4 ---
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # --- block 5 ---
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        # --- fine block ---
        x = Flatten(name='flatten')(x)
        x = Dense(1024, activation='relu', name='fc_1')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation='relu', name='fc2')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)

        combined = Concatenate(axis=1)([x, c_1_pred])

        fine_pred = Dense(self.num_classes, activation='softmax', name='predictions')(combined)

        model = Model(inputs=img_input, outputs=[c_1_pred, fine_pred], name='beecnn')
        model.summary()

        model.compile(loss='categorical_crossentropy',
                      optimizer=self.optimizer(),
                      loss_weights=[self.alpha, self.beta],
                      # optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy', 'top_k_categorical_accuracy'])
        return model
