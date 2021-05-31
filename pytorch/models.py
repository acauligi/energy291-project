import torch
import numpy as np

class FFNet(torch.nn.Module):
    """Simple class to implement a feed-forward neural network in PyTorch.
    
    Attributes:
        layers: list of torch.nn.Linear layers to be applied in forward pass.
        activation: activation function to be applied between layers.
    
    """
    def __init__(self,shape,activation=None):
        """Constructor for FFNet.
        
        Arguments:
            shape: list of ints describing network shape, including input & output size.
            activation: a torch.nn function specifying the network activation.
        """
        super(FFNet, self).__init__()
        self.shape = shape
        self.layers = []
        self.activation = activation ##TODO(pculbertson): make it possible use >1 activation... maybe? who cares
        for ii in range(0,len(shape)-1):
            self.layers.append(torch.nn.Linear(shape[ii],shape[ii+1]))

        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self, x):
        "Performs a forward pass on x, a numpy array of size (-1,shape[0])"
        for ii in range(0,len(self.layers)-1):
            x = self.layers[ii](x)
            if self.activation:
              x = self.activation(x)

        return self.layers[-1](x)