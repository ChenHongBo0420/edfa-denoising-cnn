import torch
from torch import nn
from torch import Tensor
import torch.nn.init as nn_init
import math

class TransTabWordEmbedding(nn.Module):
    r'''
    Encode tokens drawn from column names, categorical and binary features.
    '''
    def __init__(self,
        vocab_size,
        hidden_dim,
        padding_idx=0,
        hidden_dropout_prob=0,
        layer_norm_eps=1e-5,
        ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        nn_init.kaiming_normal_(self.word_embeddings.weight)
        self.norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids) -> Tensor:
        embeddings = self.word_embeddings(input_ids)
        embeddings = self.norm(embeddings)
        embeddings =  self.dropout(embeddings)
        return embeddings

class TransTabNumEmbedding(nn.Module):
    r'''
    Encode tokens drawn from column names and the corresponding numerical features.
    '''
    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.num_bias = nn.Parameter(Tensor(1, 1, hidden_dim)) # add bias
        nn_init.uniform_(self.num_bias, a=-1/math.sqrt(hidden_dim), b=1/math.sqrt(hidden_dim))

    def forward(self, num_col_emb, x_num_ts, num_mask=None) -> Tensor:
        '''args:
        num_col_emb: numerical column embedding, (# numerical columns, emb_dim)
        x_num_ts: numerical features, (bs, emb_dim)
        num_mask: the mask for NaN numerical features, (bs, # numerical columns)
        '''
        num_col_emb = num_col_emb.unsqueeze(0).expand((x_num_ts.shape[0],-1,-1))
        num_feat_emb = num_col_emb * x_num_ts.unsqueeze(-1).float() + self.num_bias
        return num_feat_emb

class Shrinkage(nn.Module):
    def __init__(self, channel):
        super(Shrinkage, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.BatchNorm1d(channel),
            nn.Linear(channel, channel),
            nn.Sigmoid(),
        )
        self.softmax = nn.Softmax(-1)
    def forward(self, x):
        x_raw = x
        x = torch.abs(x)
        x_abs = x

        x = self.gap(x)
        x = torch.flatten(x, 1)
        average = torch.mean(x, dim=1, keepdim=True)  #CS
        x = self.fc(x)
        x = torch.mul(average, x)
        x = x.unsqueeze(2)

        # g1 = x
        # g2 = x.permute(0, 2, 1).contiguous()
        # g3 = g1.matmul(g2)
        # G3 = g2.matmul(g1)
        # G = self.softmax(G3)
        # g = self.softmax(g3)
        # y = x.matmul(G)
        # x = g.matmul(x)
        # x = x + y

        sub = x_abs - x
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x_raw), n_sub)
        return x

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.shrinkage = Shrinkage(out_channels)
        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv1d(in_channels, out_channels * BasicBlock.expansion, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels * BasicBlock.expansion),
            self.shrinkage
        )
        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
         return self.residual_function(x)

class RSNet(nn.Module):

    def __init__(self, block, num_classes=83):
        super().__init__()

        self.in_channels = 128
        self.conv0_x = nn.Conv1d(1, 128, kernel_size=3, padding=1, bias=False)
        self.conv1_x = nn.Conv1d(64, 64, kernel_size=3, padding=1, bias=False)
        self.conv2_x = self._make_layer(block, 128, 1, 2)
        self.conv3_x = self._make_layer(block, 256, 1, 2)
        self.conv4_x = self._make_layer(block, 512, 1, 2)
        self.fc = nn.Linear(5632 * block.expansion, num_classes)


    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make rsnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual shrinkage block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a rsnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv0_x(x.unsqueeze(1))
        output = self.conv2_x(x)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output
def NN():
    """ return a RSNet 18 object
    """
    return RSNet(BasicBlock)




