import torch.nn as nn
import torch.nn.functional as F

class Coder(nn.Module):

    def __init__(self, n_in, n_out, n_hid=[32], output_activation=nn.Sigmoid):
        super().__init__()

        if type(n_hid) != list:
            n_hid = [n_hid]
        n_layers = [n_in] + n_hid + [n_out]

        self.layers = []
        for i_layer, (n1, n2) in enumerate(zip(n_layers, n_layers[1:])):
            mods = [nn.Linear(n1, n2, bias=True)]
            act_fn = nn.ReLU if i_layer < len(n_layers) - 2 else output_activation
            if act_fn is not None:
                mods.append(act_fn())
            layer = nn.Sequential(*mods)
            self.layers.append(layer)

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class nnNorm(nn.Module):

    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, dim=self.dim)


class Decoder(Coder):

    def __init__(self, n_in, n_out, n_hid=[32]):
        super().__init__(n_in, n_out, n_hid, output_activation=nn.Sigmoid)


class Encoder(Coder):

    def __init__(self, n_in, n_out, n_hid=[32]):
        super().__init__(n_in, n_out, n_hid, output_activation=nnNorm)