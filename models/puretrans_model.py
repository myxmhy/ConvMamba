from torch import nn


class puretrans_model(nn.Module):
    r"""
    pureTransformer
    """
    def __init__(self, num_features, sequence_length, num_classes, nhead=4, 
                 num_encoder_layers=3, num_decoder_layers = 1, dim_feedforward = 2048):
        super(puretrans_model, self).__init__()

        self.transformer = nn.Transformer(d_model=num_features, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers, 
                                          num_decoder_layers = num_decoder_layers,
                                          dim_feedforward = dim_feedforward,
                                          batch_first=True)
        self.fc = nn.Linear(sequence_length, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.transformer(x, x)
        x = x.mean(dim=2)
        x = self.fc(x)  
        return x
