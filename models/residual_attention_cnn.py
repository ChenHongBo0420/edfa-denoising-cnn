import torch
from torch import nn


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
        self._initialize_weights()

    def forward(self, x):
        x_raw = x
        x = torch.abs(x)
        x_abs = x
        x = self.gap(x)
        x = torch.flatten(x, 1)
        average = torch.mean(x, dim=1, keepdim=True)  # CS
        x = self.fc(x)
        x = torch.mul(average, x)
        x = x.unsqueeze(2)
        sub = x_abs - x
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x_raw), n_sub)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.shrinkage = Shrinkage(out_channels)
        self.attention = sa_layer(out_channels, groups=1)
        self.residual_function = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            self.shrinkage,
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * BasicBlock.expansion)
            )
        self._initialize_weights()

    def forward(self, x):
        out = self.residual_function(x)
        out = self.attention(out)
        return nn.GELU()(out + self.shortcut(x))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class sa_layer(nn.Module):
    def __init__(self, in_channels, groups):
        super(sa_layer, self).__init__()
        self.groups = groups
        self.in_channels = in_channels
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.cweight = nn.Parameter(torch.zeros(1, in_channels // (2 * groups), 1))
        self.cbias = nn.Parameter(torch.ones(1, in_channels // (2 * groups), 1))
        self.sweight = nn.Parameter(torch.zeros(1, in_channels // (2 * groups), 1))
        self.sbias = nn.Parameter(torch.ones(1, in_channels // (2 * groups), 1))
        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(in_channels // (2 * groups), in_channels // (2 * groups))

        nn.init.kaiming_normal_(self.cweight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.cbias, 0)
        nn.init.kaiming_normal_(self.sweight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.sbias, 0)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, t = x.shape
        x = x.reshape(b, groups, -1, t)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(b, -1, t)
        return x

    def forward(self, x):
        b, c, t = x.shape
        x = x.reshape(b * self.groups, -1, t)
        x_0, x_1 = x.chunk(2, dim=1)
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, t)
        out = self.channel_shuffle(out, self.groups)
        return out


class RSNet(nn.Module):
    def __init__(self, block, num_classes=83):
        super().__init__()
        self.in_channels = 32
        self.conv1_x = nn.Conv1d(1, 32, kernel_size=3, padding=1, bias=False)
        self.conv2_x = self._make_layer(block, 64, 1, 2)
        self.conv3_x = self._make_layer(block, 128, 1, 2)
        self.conv4_x = self._make_layer(block, 256, 1, 2)
        self.avg_pool = nn.AdaptiveAvgPool1d((1))
        self.fc = nn.Linear(2816 * block.expansion, num_classes)
        # self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1_x(x)
        output = self.conv2_x(x)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def NN():
    return RSNet(BasicBlock)

