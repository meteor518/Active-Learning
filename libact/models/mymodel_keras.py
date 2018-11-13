import numpy as np
from keras.layers import *
from keras.models import Model

from libact.base.interfaces import ProbabilisticModel
from keras.utils.np_utils import to_categorical
K.set_image_dim_ordering("th")

class MyModel(ProbabilisticModel):

    """读取 keras 模型结构
    """

    def __init__(self, model_folder, input_shape=(3, 224, 224), nclasses=3):
        # config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = gpu_use
        # config.gpu_options.allow_growth = True
        # sess = tf.Session(config=config)
        # KTF.set_session(sess)
        self.nclasses = nclasses
        self.model = self.model(input_shape=input_shape, nclasses=self.nclasses)
        # self.model.summary()
        self.model.load_weights(model_folder)

    def model(self, input_shape=(3, 224, 224), nclasses=3):
        """
            This function compiles and returns a Keras model.
        """
        inputs = Input(input_shape)

        pad1_1 = ZeroPadding2D(padding=(1, 1))(inputs)
        conv1_1 = Conv2D(64, 3, 3, activation='relu', name='conv1_1')(pad1_1)
        pad1_2 = ZeroPadding2D(padding=(1, 1))(conv1_1)
        conv1_2 = Conv2D(64, 3, 3, activation='relu', name='conv1_2')(pad1_2)
        pool1 = MaxPooling2D((2, 2), strides=(2, 2))(conv1_2)

        pad2_1 = ZeroPadding2D((1, 1))(pool1)
        conv2_1 = Conv2D(128, 3, 3, activation='relu', name='conv2_1')(pad2_1)
        pad2_2 = ZeroPadding2D((1, 1))(conv2_1)
        conv2_2 = Conv2D(128, 3, 3, activation='relu', name='conv2_2')(pad2_2)
        pool2 = MaxPooling2D((2, 2), strides=(2, 2))(conv2_2)

        pad3_1 = ZeroPadding2D((1, 1))(pool2)
        conv3_1 = Conv2D(256, 3, 3, activation='relu', name='conv3_1')(pad3_1)
        pad3_2 = ZeroPadding2D((1, 1))(conv3_1)
        conv3_2 = Conv2D(256, 3, 3, activation='relu', name='conv3_2')(pad3_2)
        pad3_3 = ZeroPadding2D((1, 1))(conv3_2)
        conv3_3 = Conv2D(256, 3, 3, activation='relu', name='conv3_3')(pad3_3)
        pool3 = MaxPooling2D((2, 2), strides=(2, 2))(conv3_3)

        pad4_1 = ZeroPadding2D((1, 1))(pool3)
        conv4_1 = Conv2D(512, 3, 3, activation='relu', name='conv4_1')(pad4_1)
        pad4_2 = ZeroPadding2D((1, 1))(conv4_1)
        conv4_2 = Conv2D(512, 3, 3, activation='relu', name='conv4_2')(pad4_2)
        pad4_3 = ZeroPadding2D((1, 1))(conv4_2)
        conv4_3 = Conv2D(512, 3, 3, activation='relu', name='conv4_3')(pad4_3)
        pool4 = MaxPooling2D((2, 2), strides=(2, 2))(conv4_3)

        pad5_1 = ZeroPadding2D((1, 1))(pool4)
        conv5_1 = Conv2D(512, 3, 3, activation='relu', name='conv5_1')(pad5_1)
        pad5_2 = ZeroPadding2D((1, 1))(conv5_1)
        conv5_2 = Conv2D(512, 3, 3, activation='relu', name='conv5_2')(pad5_2)
        pad5_3 = ZeroPadding2D((1, 1))(conv5_2)
        conv5_3 = Conv2D(512, 3, 3, activation='relu', name='conv5_3')(pad5_3)
        pool5 = MaxPooling2D((2, 2), strides=(2, 2))(conv5_3)

        # fc5 = base_model.layers[-8].output
        fc6 = Flatten()(pool5)
        fc7_1 = Dense(256, activation='relu', name='fc7_1')(fc6)
        dropout7_1 = Dropout(0.3, name='dropout7_1')(fc7_1)
        fc7_2 = Dense(128, activation='relu', name='fc7_2')(dropout7_1)
        # dropout7_2 = Dropout(0.2, name='dropout7_2')(fc7_2)
        # fc7_3 = Dense(128, activation="relu", name="fc7_pre3")(dropout7_2)
        prediction = Dense(nclasses, activation='softmax')(fc7_2)
        model = Model(inputs, prediction)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self, dataset, *args, **kwargs):
        x, y = dataset.format_sklearn()
        y = to_categorical(y, num_classes=self.nclasses)
        return self.model.fit(x, y, *args, **kwargs)

    def predict(self, feature, *args, **kwargs):

        return self.model.predict(feature, *args, **kwargs)

    def score(self, testing_dataset, *args, **kwargs):
        x, y = testing_dataset.format_sklearn()
        y = to_categorical(y, num_classes=self.nclasses)
        return self.model.evaluate(x, y, **kwargs)

    def predict_proba(self, feature, *args, **kwargs):
        return self.predict(feature, *args, **kwargs)

