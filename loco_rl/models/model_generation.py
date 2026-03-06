from loco_rl.models import MLP, RNN, CNN2d, CNN2dHead, ModelCfg

def generate_model(
    input_dim:int,
    output_dim:int,
    cfg:ModelCfg):
    model_type = cfg.model_type
    if model_type == "MLP":
        return MLP(input_dim, cfg.hidden_dims, output_dim, cfg.activation, cfg.final_layer_activation)
    elif model_type == "RNN":
        return RNN(input_dim, cfg.hidden_dims, output_dim, cfg.activation,
                    cfg.rnn_type, cfg.rnn_hidden_size, cfg.rnn_num_layers)
    elif model_type ==  "CNN2d":
        return CNN2d(input_dim, cfg.cnn_channels, cfg.cnn_kernel_size, cfg.cnn_stride, cfg.cnn_padding,
                     cfg.cnn_nonlinearity, cfg.cnn_use_maxpool, cfg.cnn_normlayer)
    elif model_type == "CNN2dHead":
        return CNN2dHead(
            cfg.img_shape,
            cfg.cnn_channels, cfg.cnn_kernel_size, cfg.cnn_stride, cfg.cnn_padding,
            cfg.hidden_dims, output_dim, cfg.cnn_nonlinearity, cfg.cnn_use_maxpool, cfg.cnn_normlayer)
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")



