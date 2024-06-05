from torch import nn
from .layers import MHA, MHAConfig


class easypureMHA_model(nn.Module):
    r"""
    toooooo~~ easy pureMHA......
    """
    def __init__(self, num_features, squence_length, num_classes, n_layers, nhead=1):
        super(easypureMHA_model, self).__init__()


        config = MHAConfig(d_model=num_features, 
                             n_layers=n_layers,
                             nhead = nhead)
        
        self.MHA = MHA(config)

        self.fc = nn.Linear(squence_length, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.MHA(x)
        x = x.mean(dim=2)
        x = self.fc(x)  
        return x

