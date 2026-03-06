import torch.nn as nn
from .mlp import MLP
from .activation import get_activation


def conv2d_output_shape(h, w, kernel_size=1, stride=1, padding=0, dilation=1):
    kh, kw = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, ) * 2
    sh, sw = stride if isinstance(stride, tuple) else (stride, ) * 2
    ph, pw = padding if isinstance(padding, tuple) else (padding, ) * 2
    d = dilation
    h = (h + (2 * ph) - (d * (kh - 1)) - 1) // sh + 1
    w = (w + (2 * pw) - (d * (kw - 1)) - 1) // sw + 1
    return h, w


class CNN2d(nn.Module):
    def __init__(
        self,
        in_channels=2,
        channels=(2, 4, 8),
        kernel_sizes=(5, 4, 3),
        strides=(2, 1, 1),
        paddings=None,
        nonlinearity="relu",  # Module, not Functional.
        use_maxpool=True,  # if True: convs use stride 1, maxpool downsample.
        normlayer=None,  # If None, will not be used
        ):
        super().__init__()
        paddings = [0 for _ in range(len(channels))] if paddings is None else paddings
        nonlinearity_func = get_activation(nonlinearity)
        normlayer = getattr(nn, normlayer) if isinstance(normlayer, str) else normlayer
        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings)
        in_channels = [in_channels] + list(channels)[:-1]
        ones = [1 for _ in range(len(strides))]
        if use_maxpool:
            maxp_strides = strides
            strides = ones
        else:
            maxp_strides = ones
        conv_layers = [nn.Conv2d(in_channels=ic, out_channels=oc, kernel_size=k, stride=s, padding=p)
        for (ic, oc, k, s, p) in zip(in_channels, channels, kernel_sizes, strides, paddings)]
        sequence = list()
        for conv_layer, oc, maxp_stride in zip(conv_layers, channels, maxp_strides):
            if normlayer is not None:
                sequence.extend([conv_layer, normlayer(oc), nonlinearity_func])
            else:
                sequence.extend([conv_layer, nonlinearity_func])
            if maxp_stride > 1:
                sequence.append(nn.MaxPool2d(maxp_stride))  # No padding.
        self.conv = nn.Sequential(*sequence)

    def forward(self, input_tensor):
        """Computes the convolution stack on the input; assumes correct shape already: [B,C,H,W]."""
        return self.conv(input_tensor)

    def conv_out_size(self, h, w, c=None):
        """Helper function ot return the output size for a given input shape,
        without actually performing a forward pass through the model."""
        for child in self.conv.children():
            try:
                h, w = conv2d_output_shape(h, w, child.kernel_size,
                    child.stride, child.padding)
            except AttributeError:
                pass  # Not a conv or maxpool layer.
            try:
                c = child.out_channels
            except AttributeError:
                pass  # Not a conv layer.
        return h * w * c

    def reset(self, dones=None):
        pass


class CNN2dHead(nn.Module):
    """Model component composed of a ``Conv2dModel`` component followed by 
    a fully-connected ``MlpModel`` head.  Requires full input image shape to
    instantiate the MLP head.
    """

    def __init__(
            self,
            image_shape,
            channels=(2, 4, 8),
            kernel_sizes=(5, 4, 3),
            strides=(2, 1, 1),
            paddings=None,
            hidden_sizes=None,
            output_size=None,  # if None, will not be used
            nonlinearity="relu",
            use_maxpool=False,
            normlayer= None, # if None, will not be used
            ):
        super().__init__()
        c, h, w = image_shape
        self.conv = CNN2d(
            in_channels=c,
            channels=channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
            nonlinearity=nonlinearity,
            use_maxpool=use_maxpool,
            normlayer= normlayer, # if None, will not be used
        )
        conv_out_size = self.conv.conv_out_size(h, w)
        if hidden_sizes or output_size:
            self.head = MLP(conv_out_size, hidden_sizes, output_size, activation=nonlinearity)
            if output_size is not None:
                self._output_size = output_size
            else:
                self._output_size = (hidden_sizes if
                    isinstance(hidden_sizes, int) else hidden_sizes[-1])
        else:
            self.head = lambda x: x
            self._output_size = conv_out_size

    def forward(self, input):
        """Compute the convolution and fully connected head on the input;
        assumes correct input shape: [B,C,H,W]."""
        return self.head(self.conv(input).view(input.shape[0], -1))

    @property
    def output_size(self):
        """Returns the final output size after MLP head."""
        return self._output_size


    def reset(self, dones=None):
        pass











