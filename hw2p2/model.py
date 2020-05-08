import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(IdentityBlock, self).__init__()
        self.net = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
                        nn.BatchNorm2d(out_channels))
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                                nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
                                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        x = self.shortcut(x) + self.net(x)
        return F.relu(x, inplace=True)


class ResNet(nn.Module):
    def __init__(self, feat_dim=2):
        super(ResNet, self).__init__()
        self.net = nn.Sequential(
                        nn.Conv2d(3, 128, 3, 1, 1, bias=False),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True),
                        IdentityBlock(128, 128, 1),
                        IdentityBlock(128, 256, 2),
                        IdentityBlock(256, 256, 1),
                        IdentityBlock(256, 512, 1),
                        IdentityBlock(512, 512, 1),
                        IdentityBlock(512, 1024, 2),
                        IdentityBlock(1024, 1024, 1))
        self.labels = nn.Linear(1024, 2300, bias=True)
        self.features = nn.Linear(1024, 2048, bias=True)

        self.linear_loss = nn.Linear(1024, feat_dim, bias=True)
        self.relu_loss = nn.ReLU(inplace=True)

    def forward(self, x, evalMode=False):
        output = self.net(x)

        output = F.avg_pool2d(output, [output.size(2), output.size(3)], stride=1)
        output = output.reshape(output.shape[0], output.shape[1])

        features = self.features(output)
        labels = self.labels(output)
        labels = labels / torch.norm(self.linear_label.weight, dim=1)

        loss = self.linear_loss(output)
        loss = self.relu_loss(loss)

        return loss, labels, features

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = stride == 1 and in_channel == out_channel
        hidden_size = int(in_channel * expand_ratio)

        self.bottleneck = nn.Sequential(
                            nn.Conv2d(in_channel, hidden_size, 1, 1, 0, bias=False),
                            nn.BatchNorm2d(hidden_size),
                            nn.ReLU6(inplace=True),
                            nn.Conv2d(hidden_size, hidden_size, 3, stride, 1, groups=hidden_size, bias=False),
                            nn.BatchNorm2d(hidden_size),
                            nn.ReLU6(inplace=True),
                            nn.Conv2d(hidden_size, out_channel, 1, 1, 0, bias=False),
                            nn.BatchNorm2d(out_channel))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.bottleneck(x)
        else:
            return self.bottleneck(x)


class MobileNet(nn.Module):
    def __init__(self, num_classes=2300):
        super(MobileNet, self).__init__()
        in_channel = 32
        last_channel = 4096
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 1],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        self.p = 0.2  # dropout rate, adjust!

        layers = [nn.Sequential(
                    nn.Conv2d(3, in_channel, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(in_channel),
                    nn.Dropout2d(self.p, inplace=True),  # adjust
                    nn.ReLU6(inplace=True)
                )]
        for t, c, n, s in interverted_residual_setting:
            out_channel = c
            for i in range(n):
                layers.append(InvertedResidual(in_channel, out_channel, s if i == 0 else 1, t))
                in_channel = out_channel
        layers.append(nn.Sequential(
                        nn.Conv2d(in_channel, last_channel, 1, 1, 0, bias=False),
                        nn.BatchNorm2d(last_channel),
                        nn.ReLU6(inplace=True)
                    ))
        layers.append(nn.Sequential(
                        nn.AvgPool2d(4),
                        nn.Dropout2d(self.p, inplace=True)
                    ))
        self.net = nn.Sequential(*layers)
        self.classifier = nn.Conv2d(last_channel, num_classes, 1, 1, 0, bias=True)
        self.initialize_weights()

    def forward(self, x):
        x_features = self.net(x)  # batch_size x 1280 x 1 x 1
        x_scores = self.classifier(x_features)  # batch_size x 2300 x 1 x 1
        return x_features.view(x.size(0), -1), x_scores.view(x.size(0), -1)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
