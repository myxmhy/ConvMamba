from torch import nn


class pureGRU_model(nn.Module):
    r"""
    pureGRU
    """
    def __init__(self, num_features, num_classes, hidden_size=512, num_layers=2):
        super(pureGRU_model, self).__init__()

        self.GRU = nn.GRU(input_size=num_features,
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x,_ = self.GRU(x)
        x = x.mean(dim=1)
        x = self.fc(x)  
        return x
