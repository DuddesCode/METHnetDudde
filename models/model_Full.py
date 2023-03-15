"""class containing a model that takes two models andsets them in a sequential order."""
import torch.nn as nn

class Partial_Net(nn.Module):
    def __init__(self, model_enc, model_dec) -> None:
        super(Partial_Net, self).__init__()
        self.alpha = model_enc
        self.beta = model_dec

    def forward(self, x):
        x = self.alpha(x)
        x = self.beta(x)
        return x

    def getEncoder(self):
        return self.alpha

    def getDecoder(self):
        return self.beta

