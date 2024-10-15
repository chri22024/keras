from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Dense, MaxPooling2D
from tensorflow.keras.utils import plot_model


def add_conv_pool_layers(x, filters, kernel_size, pool_size, activation='relu'):
    x = Conv2D(filters, kernel_size, padding='same', use_bias = False)(x)
    x = Activation(activation)(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, kernel_size, padding='same', activation=activation)(x)
    x = Activation(activation)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D(pool_size)(x)

    return x



def add_dense_layer(x, dim, use_bn = True, activation='relu'):
    x = Dense(dim, use_bias = not use_bn)(x)
    x = Activation(activation)(x)

    if use_bn:
        x = BatchNormalization()(x)

    return x


def save_model_info(intfo_file, graph_file, model):
    with open(intfo_file, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    plot_model(model, to_file=graph_file, show_shapes=True)

