from isaaclab.utils import configclass


@configclass
class ModelCfg:
    # MLP specific
    model_type: str = "MLP"
    hidden_dims: list[int] = [512, 256, 128]
    activation: str = "elu"
    final_layer_activation = None

    # RNN specific
    rnn_type = "gru"
    rnn_hidden_size = 256
    rnn_num_layers = 1

    # CNN specific
    img_shape = (2, 17, 13)
    cnn_channels = (24, 24, 24)
    cnn_kernel_size = (4, 3, 2)
    cnn_stride = (2, 1, 1)
    cnn_nonlinearity = "relu"
    cnn_padding = None
    cnn_use_maxpool = True
    cnn_normlayer = None