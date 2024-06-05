from torch import nn
import numpy as np
from .layers import Mamba, MambaConfig
import math
class easymamba_model(nn.Module):
    r"""
    toooooo~~ easy Mamba......
    """
    def __init__(self, num_features, num_classes, mamba_layers, d_state=16, d_conv=3, expand = 2, 
                 num_conv_layer = 6, conv_outchannel=512, conv_strid_size=128):
        super(easymamba_model, self).__init__()

        self.enc = Conv_Encoder(num_features, num_conv_layer, conv_outchannel, conv_strid_size)

        config = MambaConfig(d_model=conv_outchannel, 
                             n_layers=mamba_layers,
                             d_state = d_state,
                             expand_factor = expand,
                             d_conv = d_conv)
        
        self.mamba = Mamba(config)

        self.fc = nn.Linear(conv_outchannel, num_classes)

    def forward(self, x):
        x = self.enc(x)
        x = x.permute(0, 2, 1)
        x = self.mamba(x)
        x = x.mean(dim=1)
        x = self.fc(x)  
        return x
    
class Conv_Encoder(nn.Module):
    """Conv Encoder"""

    def __init__(self, num_features, num_conv_layer, conv_outchannel, conv_strid_size):
        super(Conv_Encoder, self).__init__()
        channel_step = math.ceil(conv_outchannel / 2**(num_conv_layer-1))
        out_channel_list = [channel_step * 2**(i) for i in range(num_conv_layer-1)]
        out_channel_list += [conv_outchannel]

        strid_step = np.log2(conv_strid_size)
        strid_step_layer = int(strid_step / num_conv_layer)
        strid_list = [2**strid_step_layer for _ in range(num_conv_layer-1)]
        strid_list = [conv_strid_size // 2**(strid_step_layer*(num_conv_layer-1))] + strid_list

        strid_list = self.factorize_into_powers_of_two(conv_strid_size, num_conv_layer)
        self.enc = nn.Sequential(
              nn.Conv1d(in_channels=num_features, out_channels=out_channel_list[0], kernel_size=15, stride=strid_list[0],
                                 padding=7),
            *[nn.Conv1d(in_channels=out_channel_list[i-1], out_channels=out_channel_list[i], kernel_size=15,
                     stride=strid_list[i], padding=7) for i in range(1,num_conv_layer-1)]
        )
        if num_conv_layer > 1:
            self.enc =  nn.Sequential(self.enc,
                    nn.Conv1d(in_channels=out_channel_list[-2], out_channels=out_channel_list[-1], kernel_size=15, stride=strid_list[-1],
                                 padding=7))
            
    def factorize_into_powers_of_two(self, n, parts):
        def helper(n, parts):
            if parts == 1:
                return [n]
            part_value = 2 ** (n.bit_length() // parts)
            remaining = n // part_value
            return [part_value] + helper(remaining, parts - 1)

        factors = helper(n, parts)
        # Adjust to make sure all factors are powers of 2
        for i in range(len(factors)):
            factors[i] = 2 ** (factors[i].bit_length() - 1)
        factors.sort(reverse=True)
        return factors

    def forward(self, x):
        latent = x
        for i in range(len(self.enc)):
            latent = self.enc[i](latent)
        return latent