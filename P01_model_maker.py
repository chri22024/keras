
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import P10_util as util
import P11_model_util as mutil

class ModelMaker:
    

    def __init__(self,  dst_dir, est_file, info_file, graph_file, hist_file, dims, lr,  batch_size, epochs, valid_rate):
        self.dst_dir = dst_dir
        self.est_file = est_file
        self.info_file = info_file
        self.graph_file =graph_file
        self.hist_file = hist_file
        self.dims = dims
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.vaild_rata = vaild_rate


    def define_model(self):








        for dim in self.dense_dims[1:-1]:
            x = mutil.add_dense_layer(x, dim)


        x = mutil.add_dense_layer(
            x, self.dims[-1], use_bn=False, activation='softmax'
        )

        model = Model(input_x, x)

        model.compile(
            optimizer = Adam(learning_rate = self.lr),
            loss = 'categorical_crossentropy',
            metrics=['accuracy']
        )

        return model





    
    def fit_model(self):


        train_data, train_classes = util.load_data()


        model = self.define_model()

        history = model.fit(
            train_data,
            train_classes,
            batch_size = self.batch_size,
            epochs = self.epochs,
            validation_split = self.vaild_rata
        )


        return model, history.history
    

    def execute(self):

        model, history = self.fit_model()

        util.mkdir(self.dst_dir, rm=True)
        model.save(self.est_file)

        mutil.save_model_info(self.info_file, self.graph_file, model)




        util.plot(history, self.hist_file)




        print('val_loss: %f' % history['val_loss'][-1])
