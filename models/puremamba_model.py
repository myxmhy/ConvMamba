from torch import nn
from .layers import Mamba, MambaConfig

class puremamba_model(nn.Module):
    r"""
    pureMamba
    """
    def __init__(self, num_features, sequence_length, num_classes, 
                 mamba_layers, d_state=16, d_conv=3, expand = 2):
        super(puremamba_model, self).__init__()

        config = MambaConfig(d_model=num_features, 
                             n_layers=mamba_layers,
                             d_state = d_state,
                             expand_factor = expand,
                             d_conv = d_conv)
        
        self.mamba = Mamba(config)

        self.fc = nn.Linear(sequence_length, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.mamba(x)
        x = x.mean(dim=2)
        x = self.fc(x)  
        return x
 