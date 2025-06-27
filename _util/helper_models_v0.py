from _util.util_v1 import *
import _util.util_v1 as uutil
from _util.pytorch_v1 import *
import _util.pytorch_v1 as utorch
from _util.twodee_v0 import *
import _util.twodee_v0 as u2d


class LambdaLayer(nn.Module):
    def __init__(self, func):
        super(LambdaLayer, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class AbsoluteValue(nn.Module):
    def __init__(self):
        super(AbsoluteValue, self).__init__()
        return

    def forward(self, x):
        return torch.abs(x)


class MaskedImageLossL1(nn.Module):
    def __init__(self):
        super(MaskedImageLossL1, self).__init__()
        return

    def forward(self, alpha, gt, pred):
        return (alpha * (gt - pred).abs().mean(dim=1, keepdim=True)).mean()


class PositionalEncoding(nn.Module):
    def __init__(self, phases, freq_min=1, freq_max=10000):
        super(PositionalEncoding, self).__init__()
        self.phases = phases
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.size = 2 * phases  # for sin and cos
        ls = np.linspace(0, 1, phases)
        mult = torch.tensor(
            np.exp(ls * np.log(freq_max) + (1 - ls) * np.log(freq_min))
        ) * (2 * np.pi)
        self.register_buffer("multiplier", mult)
        return

    def forward(self, x):
        t = x.unsqueeze(-1) * self.multiplier
        t = torch.stack(
            [
                torch.sin(t),
                torch.cos(t),
            ],
            dim=-1,
        ).flatten(-2)
        return t.float()


class ResBlock(nn.Module):
    def __init__(
        self,
        depth,
        channels,
        kernel,
        channels_in=None,  # in case different from channels
        activation=nn.ReLU,
        normalization=nn.BatchNorm2d,
    ):
        # activation()
        # normalization(channels)
        super(ResBlock, self).__init__()
        self.depth = depth
        self.channels = channels
        self.channels_in = channels_in
        self.kernel = kernel
        self.activation = activation
        self.normalization = normalization

        # create sequential network
        od = OrderedDict()
        for i in range(depth):
            chin = channels_in if channels_in is not None and i == 0 else channels
            od[f"conv{i}"] = nn.Conv2d(
                chin,
                channels,
                kernel_size=kernel,
                padding=kernel // 2,
                bias=True,
                padding_mode="replicate",
            )
            if activation is not None:
                od[f"act{i}"] = activation()
            if normalization is not None:
                od[f"norm{i}"] = normalization(channels)
        self.net = nn.Sequential(od)

        # last activation/normalization
        od_tail = OrderedDict()
        if activation is not None:
            od_tail[f"act{depth}"] = activation()
        if normalization is not None:
            od_tail[f"norm{depth}"] = normalization(channels)
        self.net_tail = nn.Sequential(od_tail)
        return

    def forward(self, x):
        if self.channels_in is None:
            return self.net_tail(x + self.net(x))
        else:
            head = self.net[0](x)
            t = head
            for body in self.net[1:]:
                t = body(t)
            return self.net_tail(head + t)


class Interpolator2d(nn.Module):
    def __init__(self, size=None, mode="nearest"):
        # will work as long as batch dim matches
        # modes: nearest, bilinear, bicubic, area (downscaling)
        super(Interpolator2d, self).__init__()
        self.size = (size, size) if type(size) == int else size
        self.mode = mode
        return

    def forward(self, x, size=None, mode=None):
        # local vars override defaults
        size = size or self.size
        size = (size, size) if type(size) == int else size
        mode = mode or self.mode
        return torch.cat(
            [
                (
                    TF.interpolate(
                        t,
                        size=size,
                        mode=mode,
                        # align_corners=not mode in ['nearest', 'area'],
                    )
                    if t.shape[-2:] != self.size
                    else t
                )
                for t in x
                if t is not None
            ],
            dim=1,
        )


class Injector(nn.Module):
    def __init__(
        self,
        size,  # input image height/width
        depth,  # of the resnet
        channels_input,
        channels_preprocess,  # on input
        channels_input_aux,  # list
        channels_resnet,
        kernel,
        activation=nn.ReLU,
        normalization=nn.BatchNorm2d,
        interpolation_mode="bicubic",
    ):
        # activation()
        # normalization(channels)
        super(Injector, self).__init__()
        self.size = (size, size) if type(size) == int else size
        self.depth = depth
        self.channels_input = channels_input
        self.channels_preprocess = channels_preprocess
        self.channels_input_aux = channels_input_aux
        self.channels_resnet = channels_resnet
        self.kernel = kernel
        self.activation = activation
        self.normalization = normalization
        self.interpolation_mode = interpolation_mode

        # adds activation and normalization to tail of conv
        def _conv_and_tail(c_in, c_out):
            od = OrderedDict()
            od["conv"] = nn.Conv2d(
                c_in,
                c_out,
                kernel_size=kernel,
                padding=kernel // 2,
                padding_mode="replicate",
            )
            if activation is not None:
                od["act"] = activation()
            if normalization is not None:
                od["norm"] = normalization(c_out)
            return nn.Sequential(od)

        # create input preprocessor
        self.net_preprocessor = _conv_and_tail(
            channels_input,
            channels_preprocess,
        )

        # create aux input interpolator
        self.net_interpolator = Interpolator2d(
            size,
            mode=interpolation_mode,
        )

        # create converter to resnet channels
        self.net_resnet_converter = _conv_and_tail(
            sum(channels_input_aux) + channels_preprocess,
            channels_resnet,
        )

        # create resnet
        self.net_resnet = ResBlock(
            depth,
            channels_resnet,
            kernel,
            activation=activation,
            normalization=normalization,
        )
        return

    def forward(self, x, x_aux):
        # preprocess input and cat with aux input
        t = self.net_interpolator(
            [
                x,
            ]
        )
        t = self.net_preprocessor(t)
        t = self.net_interpolator(
            x_aux
            + [
                t,
            ]
        )

        # feed through resnet
        t = self.net_resnet_converter(t)
        t = self.net_resnet(t)
        return t


from _train.danbooru_tagger.models.kate import Model as Classifier
import torchvision as tv


class ResnetFeatureExtractor(nn.Module):
    def __init__(self, inferserve_query):
        super().__init__()
        self.inferserve_query = inferserve_query
        if self.inferserve_query == "torchvision":
            # use pytorch pretrained resnet50
            self.inferserve = None
            self.base_hparams = None
            resnet = tv.models.resnet50(pretrained=True)

            self.resize = TT.Resize(256)
            self.resnet_preprocess = TT.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
            self.conv1 = resnet.conv1
            self.bn1 = resnet.bn1
            self.relu = resnet.relu
            self.maxpool = resnet.maxpool
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3
        elif self.inferserve_query == "rf5":
            # use rf5 resnet50
            self.inferserve = None
            self.base_hparams = None
            resnet = torch.hub.load("RF5/danbooru-pretrained", "resnet50")

            self.resize = TT.Resize(256)
            self.resnet_preprocess = TT.Normalize(
                mean=[0.7137, 0.6628, 0.6519], std=[0.2970, 0.3017, 0.2979]
            )
            self.conv1 = resnet[0][0]
            self.bn1 = resnet[0][1]
            self.relu = resnet[0][2]
            self.maxpool = resnet[0][3]
            self.layer1 = resnet[0][4]
            self.layer2 = resnet[0][5]
            self.layer3 = resnet[0][6]
        else:
            # use pretrained kate, danbooru-specific
            self.inferserve = None
            base = Classifier.load_from_checkpoint(
                "./_train/danbooru_tagger/runs/waning_kate_vulcan0001/checkpoints/"
                "epoch=0022-val_f2=0.4461-val_loss=0.0766.ckpt"
            )
            self.base_hparams = base.hparams

            self.resize = TT.Resize(base.hparams.largs.danbooru_sfw.size)
            self.resnet_preprocess = base.resnet_preprocess
            self.conv1 = base.resnet.conv1
            self.bn1 = base.resnet.bn1
            self.relu = base.resnet.relu
            self.maxpool = base.resnet.maxpool
            self.layer1 = base.resnet.layer1
            self.layer2 = base.resnet.layer2
            self.layer3 = base.resnet.layer3
        return

    def forward(self, x):
        ans = {}
        x = self.resize(x)
        x = self.resnet_preprocess(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        ans["conv1"] = x
        x = self.maxpool(x)
        x = self.layer1(x)
        ans["layer1"] = x
        x = self.layer2(x)
        ans["layer2"] = x
        x = self.layer3(x)
        ans["layer3"] = x
        return ans
