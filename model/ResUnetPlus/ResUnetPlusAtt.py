import torch
import torch.nn as nn


class EdgeAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.edge_conv = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        edge = self.edge_conv(x)
        return x * self.sigmoid(edge)

class Squeeze_Excitation(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Linear(channel, channel // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // 16, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        b, c, _, _ = inputs.shape
        x = self.pool(inputs).view(b, c)
        x = self.net(x).view(b, c, 1, 1)
        x = inputs * x
        return x

class Stem_Block(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_c),
        )

        self.attn = Squeeze_Excitation(out_c)

    def forward(self, inputs):
        x = self.c1(inputs)
        s = self.c2(inputs)
        y = self.attn(x + s)
        return y

class ResNet_Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        mid_channels = out_channels // 4
        self.c1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_channels),
        )

        self.attn = Squeeze_Excitation(out_channels)

    def forward(self, inputs):
        x = self.c1(inputs)
        s = self.c2(inputs)
        y = self.attn(x + s)
        return y

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rate=[1, 6, 12, 18]):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=rate[0], padding=rate[0]),
            nn.BatchNorm2d(out_channels)
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=rate[1], padding=rate[1]),
            nn.BatchNorm2d(out_channels)
        )

        self.c3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=rate[2], padding=rate[2]),
            nn.BatchNorm2d(out_channels)
        )

        self.c4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=rate[3], padding=rate[3]),
            nn.BatchNorm2d(out_channels)
        )

        self.c5 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)


    def forward(self, inputs):
        x1 = self.c1(inputs)
        x2 = self.c2(inputs)
        x3 = self.c3(inputs)
        x4 = self.c4(inputs)
        x = x1 + x2 + x3 + x4
        y = self.c5(x)
        return y

class Attention_Block(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        out_channels = in_channels[1]

        self.g_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels[0]),
            nn.ReLU(),
            nn.Conv2d(in_channels[0], out_channels, kernel_size=3, padding=1),
            nn.MaxPool2d((2, 2))
        )

        self.x_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels[1]),
            nn.ReLU(),
            nn.Conv2d(in_channels[1], out_channels, kernel_size=3, padding=1),
        )

        self.gc_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels[1]),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, g, x):
        g_pool = self.g_conv(g)
        x_conv = self.x_conv(x)
        gc_sum = g_pool + x_conv
        gc_conv = self.gc_conv(gc_sum)
        y = gc_conv * x
        return y

class Decoder_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.a1 = Attention_Block(in_channels)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.r1 = ResNet_Block(in_channels[0] + in_channels[1], out_channels, stride=1)

    def forward(self, g, x):
        d = self.a1(g, x)
        d = self.up(d)
        d = torch.cat([d, g], axis=1)
        d = self.r1(d)
        return d

class ResUnetPlusAtt(nn.Module):
    def __init__(self):
        super().__init__()

        self.c1 = Stem_Block(1, 16, stride=1)
        self.c2 = ResNet_Block(16, 32, stride=2)
        self.c3 = ResNet_Block(32, 64, stride=2)
        self.c4 = ResNet_Block(64, 128, stride=2)
        self.c5 = ResNet_Block(128, 256, stride=2)

        self.b1 = ASPP(256, 512)

        self.d1 = Decoder_Block([128, 512], 256)
        self.d2 = Decoder_Block([64, 256], 128)
        self.d3 = Decoder_Block([32, 128], 64)
        self.d4 = Decoder_Block([16, 64], 32)

        self.edge_attn1 = EdgeAttention(256)
        self.edge_attn2 = EdgeAttention(128)
        self.edge_attn3 = EdgeAttention(64)
        self.edge_attn4 = EdgeAttention(32)

        self.aspp = ASPP(32, 16)
        self.output = nn.Conv2d(16, 2, kernel_size=1, padding=0)

    def forward(self, inputs):
        c1 = self.c1(inputs)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        c4 = self.c4(c3)
        c5 = self.c5(c4)

        b1 = self.b1(c5)

        d1 = self.d1(c4, b1)
        d1 = self.edge_attn1(d1)

        d2 = self.d2(c3, d1)
        d2 = self.edge_attn2(d2)

        d3 = self.d3(c2, d2)
        d3 = self.edge_attn3(d3)

        d4 = self.d4(c1, d3)
        d4 = self.edge_attn4(d4)

        output = self.aspp(d4)
        output = self.output(output)

        return output



