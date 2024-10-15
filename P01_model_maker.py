from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import P10_util as util
import P11_model_util as mutil

class ModelMaker:
    

    def __init__(self, src_dir, dst_dir, est_file, info_file, graph_file, input_size, hist_file,
                 filters, dense_dims, lr, min_lr,  batch_size, epochs, vaild_rate, reuse_cnt, es_patience, lr_patience):
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.est_file = est_file
        self.info_file = info_file
        self.graph_file =graph_file
        self.hist_file = hist_file
        self.input_size = input_size
        self.dense_dims = dense_dims
        self.lr = lr
        self.min_lr = min_lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.vaild_rata = vaild_rate
        self.reuse_count = reuse_cnt
        self.es_patience = es_patience
        self.lr_patience = lr_patience


    def define_model(self):


        base_model = VGG16(include_top = False, input_shape=(*self.input_size, 3))



        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        x = Flatten()(x)


        for dim in self.dense_dims[:-1]:
            x = mutil.add_dense_layer(x, dim)

        x = mutil.add_dense_layer(x, self.dense_dims[-1] - 1, use_bn = False, activation='softmax')


        model = Model(base_model.input, x)


        model.compile(
            optimizer = Adam(learning_rate = self.lr),
            loss = 'categorical_crossentropy',
            metrics=['accuracy']
        )

        return model
    
    def fit_model(self):

        train_ds, train_n, valid_ds, valid_n = util.make_generator(
            self.src_dir, self.vaild_rata, self.input_size, self.batch_size
        )


        model = self.define_model()

        early_stopping = EarlyStopping(
            patience = self.es_patience,
            restore_best_weights = True,
            verbose = 1

        )

        reduce_lr_op = ReduceLROnPlateau(
            patience = self.lr_patience,
            min_lr = self.min_lr,
            verbose = 1
        )
        callbacks = [early_stopping, reduce_lr_op]

        history = model.fit(
            train_ds,
            steps_per_epoch = int(train_n * self.reuse_count / self.batch_size),
            epochs = self.epochs,
            validation_data = valid_ds,
            validation_steps = int(valid_n * self.reuse_count / self.batch_size),
            callbacks = callbacks
        )

        return model, history.history
    

    def execute(self):

        model, history = self.fit_model()

        util.mkdir(self.dst_dir, rm=True)
        model.save(self.est_file)
        util.plot(history, self.hist_file)

        mutil.save_model_info(self.info_file, self.graph_file, model)

        util.plot(history, self.hist_file)


        min_val = min(history['val_loss'])
        min_ind = history['val_loss'].index(min_val)
        print('val_loss: %f(Epoch: %d)' % (min_val, min_ind + 1))
