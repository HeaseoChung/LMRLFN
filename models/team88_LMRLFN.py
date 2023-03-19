import torch.nn as nn
import torch.nn.functional as F


lrelu_value = 0.1
act = nn.LeakyReLU(lrelu_value)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    """
    Activation functions for ['relu', 'lrelu', 'prelu'].

    Parameters
    ----------
    act_type: str
        one of ['relu', 'lrelu', 'prelu'].
    inplace: bool
        whether to use inplace operator.
    neg_slope: float
        slope of negative region for `lrelu` or `prelu`.
    n_prelu: int
        `num_parameters` for `prelu`.
    ----------
    """
    act_type = act_type.lower()
    if act_type == "relu":
        layer = nn.ReLU(inplace)
    elif act_type == "lrelu":
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == "prelu":
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError(
            "activation layer [{:s}] is not found".format(act_type)
        )
    return layer


def _make_pair(value):
    if isinstance(value, int):
        value = (value,) * 2
    return value


def conv_layer(in_channels, out_channels, kernel_size, bias=True):
    """
    Re-write convolution layer for adaptive `padding`.
    """
    kernel_size = _make_pair(kernel_size)
    padding = (int((kernel_size[0] - 1) / 2), int((kernel_size[1] - 1) / 2))
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=padding, bias=bias
    )


class RepConv(nn.Module):
    def __init__(self, n_feats):
        super(RepConv, self).__init__()
        self.rep_conv = nn.Conv2d(n_feats, n_feats, 3, 1, 1)

    def forward(self, x):
        out = self.rep_conv(x)

        return out


class BasicBlock(nn.Module):
    """Basic block for building HFAN

    Args:
        n_feats (int): The number of feature maps.

    Diagram:
        --RepConv--LeakyReLU--RepConv--

    """

    def __init__(self, n_feats):
        super(BasicBlock, self).__init__()
        self.conv1 = RepConv(n_feats)
        self.conv2 = RepConv(n_feats)

    def forward(self, x):
        res = self.conv1(x)
        res = act(res)
        res = self.conv2(res)

        return res


class HFAB(nn.Module):
    """High-Frequency Attention Block

    args:
        n_feats (int): The number of input feature maps.
        up_blocks (int): The number of RepConv in this HFAB.
        mid_feats (int): Input feature map numbers of RepConv.

    Diagram:
        --Reduce_dimension--[RepConv]*up_blocks--Expand_dimension--Sigmoid--

    """

    def __init__(self, n_feats, up_blocks, mid_feats):
        super(HFAB, self).__init__()
        self.squeeze = nn.Conv2d(n_feats, mid_feats, 3, 1, 1)
        convs = [BasicBlock(mid_feats) for _ in range(up_blocks)]
        self.convs = nn.Sequential(*convs)
        self.excitate = nn.Conv2d(mid_feats, n_feats, 3, 1, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = act(self.squeeze(x))
        out = act(self.convs(out))
        out = self.excitate(out)
        out = self.sigmoid(out)
        out *= x

        return out


class RLFB(nn.Module):
    def __init__(self, in_channels, mid_channels, hfab_channels, up_blocks):
        super(RLFB, self).__init__()
        self.c1_r = conv_layer(in_channels, mid_channels, 3)
        self.c2_r = conv_layer(mid_channels, mid_channels, 3)
        self.c3_r = conv_layer(mid_channels, in_channels, 3)

        self.c5 = conv_layer(in_channels, in_channels, 1)
        self.hfab = HFAB(in_channels, up_blocks, hfab_channels)
        self.act = activation("lrelu", neg_slope=0.05)

    def forward(self, x):
        out = self.c1_r(x)
        out = self.act(out)

        out = self.c2_r(out)
        out = self.act(out)

        out = self.c3_r(out)
        out = self.act(out)

        out = out + x
        out = self.hfab(self.c5(out))
        return out


class LMRLFN(nn.Module):
    def __init__(
        self,
        down_blocks=4,
        up_blocks=[2, 1, 1, 1],
        hfab_feats=16,
        n_feats=36,
        m_feats=40,
        n_colors=3,
        scale=4,
    ):
        super(LMRLFN, self).__init__()

        self.down_blocks = down_blocks
        up_blocks = up_blocks
        hfab_feats = hfab_feats
        n_feats = n_feats
        m_feats = m_feats
        n_colors = n_colors
        scale = scale

        self.head = nn.Conv2d(n_colors, n_feats, 3, 1, 1)

        self.warmup = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            HFAB(n_feats, up_blocks[0], hfab_feats - 4),
        )

        self.block_1 = RLFB(n_feats, m_feats, hfab_feats, 2)
        self.block_2 = RLFB(n_feats, m_feats, hfab_feats, 1)
        self.block_3 = RLFB(n_feats, m_feats, hfab_feats, 1)
        self.block_4 = RLFB(n_feats, m_feats, hfab_feats, 1)

        self.lr_conv = nn.Conv2d(n_feats, n_feats, 3, 1, 1)

        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, n_colors * (scale**2), 3, 1, 1),
            nn.PixelShuffle(scale),
        )

    def forward(self, x):
        x = self.head(x)

        h = self.warmup(x)

        h = self.block_1(h)
        h = self.block_2(h)
        h = self.block_3(h)
        h = self.block_4(h)

        out_low_resolution = self.lr_conv(h) + x
        output = self.tail(out_low_resolution)

        return output
