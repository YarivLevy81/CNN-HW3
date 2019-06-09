import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Block, Linear, ReLU, Sigmoid, Dropout, Sequential


class MLP(Block):
    """
    A simple multilayer perceptron model based on our custom Blocks.
    Architecture is (with ReLU activation):

        FC(in, h1) -> ReLU -> FC(h1,h2) -> ReLU -> ... -> FC(hn, num_classes)

    Where FC is a fully-connected layer and h1,...,hn are the hidden layer
    dimensions.
    If dropout is used, a dropout layer is added after every activation
    function.
    """
    def __init__(self, in_features, num_classes, hidden_features=(),
                 activation='relu', dropout=0, **kw):
        super().__init__()
        """
        Create an MLP model Block.
        :param in_features: Number of features of the input of the first layer.
        :param num_classes: Number of features of the output of the last layer.
        :param hidden_features: A sequence of hidden layer dimensions.
        :param activation: Either 'relu' or 'sigmoid', specifying which 
        activation function to use between linear layers.
        :param: Dropout probability. Zero means no dropout.
        """
        blocks = []
        self.in_features = in_features
        self.num_classes = num_classes
        self.hidden_features = hidden_features
        self.dropout = dropout

        # TODO: Build the MLP architecture as described.
        # ====== YOUR CODE: ======
        linear_kw = {'wstd': kw['wstd']} if 'wstd' in kw else {}
        
        blocks.append(Linear(in_features, hidden_features[0], **linear_kw))
        if activation =='relu':
            blocks.append(ReLU())
        else:
            blocks.append(Sigmoid())
            
        if dropout: blocks.append(Dropout(p=dropout))
            
        for i in range(len(hidden_features)-1):  
            blocks.append(Linear(hidden_features[i], hidden_features[i+1], **linear_kw))
            if activation =='relu':
                blocks.append(ReLU())
            else:
                blocks.append(Sigmoid())
            if dropout: blocks.append(Dropout(p=dropout))
        ## add last layer
        blocks.append(Linear(hidden_features[-1], num_classes, **linear_kw))


#         linear_kw = {'wstd': kw['wstd']} if 'wstd' in kw else {}
#         for Din, Dout in zip([in_features] + hidden_features, hidden_features):
#             blocks.append(Linear(Din, Dout, **linear_kw))
#             blocks.append(ReLU())
#             if dropout:
#                 blocks.append(Dropout(p=dropout))
#         blocks.append(Linear(hidden_features[-1], num_classes, **linear_kw))    
        # ========================
        self.sequence = Sequential(*blocks)


    def forward(self, x, **kw):
        return self.sequence(x, **kw)

    def backward(self, dout):
        return self.sequence.backward(dout)

    def params(self):
        return self.sequence.params()

    def train(self, training_mode=True):
        self.sequence.train(training_mode)

    def __repr__(self):
        return f'MLP, {self.sequence}'


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(Conv -> ReLU)*P -> MaxPool]*(N/P) -> (Linear -> ReLU)*M -> Linear
    """
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param filters: A list of of length N containing the number of
            filters in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        """
        super().__init__()
        self.in_size = in_size
        self.out_classes = out_classes
        self.filters = filters
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims
        self.num_pool_layers =0
        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()
        

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        # [(Conv -> ReLU)*P -> MaxPool]*(N/P)
        # Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        # Pooling to reduce dimensions.
        # ====== YOUR CODE: ======
        for i, (in_chnl, out_chnl) in enumerate(zip([in_channels] + self.filters, self.filters)):
            layers.extend([nn.Conv2d(in_chnl, out_chnl, kernel_size=3, stride=1, padding=1), nn.ReLU()])
            #add new conv2d layer
            if (i+1)%self.pool_every == 0:
                layers.append(nn.MaxPool2d(2))
                self.num_pool_layers += 1
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the classifier part of the model:
        # (Linear -> ReLU)*M -> Linear
        # You'll need to calculate the number of features first.
        # The last Linear layer should have an output dimension of out_classes.
        # ====== YOUR CODE: ======
        resize_factor = 2 ** self.num_pool_layers
        # 2 is the stride. Every time we pass through and pool-layer we got this resize factor
        number_of_features_first_layer_input = (in_h // resize_factor)*(in_w // resize_factor) * self.filters[-1]
        for layer_dim_input, layer_dim_output in zip([number_of_features_first_layer_input] + self.hidden_dims, self.hidden_dims):
            layers.extend([nn.Linear(layer_dim_input, layer_dim_output), nn.ReLU()])
            
        # Now add last layer
        layers.extend([nn.Linear(self.hidden_dims[-1], self.out_classes), nn.ReLU()])
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        # Extract features from the input, run the classifier on them and
        # return class scores.
        # ====== YOUR CODE: ======
        out = self.feature_extractor(x)
        # since the output of the feature extractor is a 4D tensor and the
        # classifier expects 2D tensor, flatten it
        out = out.view(out.size(0), -1)
        # classify
        out = self.classifier(out)
        # ========================
        return out


class YourCodeNet(ConvClassifier):
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        super().__init__(in_size, out_classes, filters, pool_every, hidden_dims)

    # TODO: Change whatever you want about the ConvClassifier to try to
    # improve it's results on CIFAR-10.
    # For example, add batchnorm, dropout, skip connections, change conv
    # filter sizes etc.
    # ====== YOUR CODE: ======
    #raise NotImplementedError()
    # ========================

