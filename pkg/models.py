import torch
import torch.nn as nn
from typing import Any, Callable, Tuple, Callable, List, Optional, Type, Union
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d
from torch import Tensor
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.resnet import conv1x1, conv3x3


class CustomResNet(nn.Module):
    def __init__(self, model):
        super(CustomResNet, self).__init__()

        self.l1 = nn.Sequential(model.conv1,
                                model.bn1,
                                model.relu,
                                model.maxpool,
                                model.layer1)
        self.l2 = nn.Sequential(
                                model.layer2)
        self.l3 = nn.Sequential(
                                model.layer3)

    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        return x1, x2, x3


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class OCBE(nn.Module):
    """One-Class Bottleneck Embedding module.
    Args:
        block (Bottleneck): Expansion value is extracted from this block.
        layers (int): Numbers of OCE layers to create after multiscale feature fusion.
        groups (int, optional): Number of blocked connections from input channels to output channels.
            Defaults to 1.
        width_per_group (int, optional): Number of layers in each intermediate convolution layer. Defaults to 64.
        norm_layer (Optional[Callable[..., nn.Module]], optional): Batch norm layer to use. Defaults to None.
    """

    def __init__(
        self,
        block: Type[Union[Bottleneck, BasicBlock]],
        layers: int,
        groups: int = 1,
        width_per_group: int = 64,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.groups = groups
        self.base_width = width_per_group
        self.inplanes = 256 * block.expansion
        self.dilation = 1
        self.bn_layer = self._make_layer(block, 512, layers, stride=2)

        self.conv1 = conv3x3(64 * block.expansion, 128 * block.expansion, 2)
        self.bn1 = norm_layer(128 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(128 * block.expansion, 256 * block.expansion, 2)
        self.bn2 = norm_layer(256 * block.expansion)
        self.conv3 = conv3x3(128 * block.expansion, 256 * block.expansion, 2)
        self.bn3 = norm_layer(256 * block.expansion)

        # This is present in the paper but not in the original code. With some initial experiments, removing this leads
        # to better results
        # self.conv4 = conv1x1(256 * block.expansion * 3, 256 * block.expansion * 3, 1)  # x3 as we concatenate 3 layers
        # self.bn4 = norm_layer(256 * block.expansion * 3)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(
        self,
        block: Type[Union[Bottleneck, BasicBlock]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes * 3, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes * 3,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, features: List[Tensor]) -> Tensor:
        """Forward-pass of Bottleneck layer.
        Args:
            features (List[Tensor]): List of features extracted from the encoder.
        Returns:
            Tensor: Output of the bottleneck layer
        """
        # Always assumes that features has length of 3
        feature0 = self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(features[0]))))))
        feature1 = self.relu(self.bn3(self.conv3(features[1])))
        feature_cat = torch.cat([feature0, feature1, features[2]], 1)
        output = self.bn_layer(feature_cat)
        #output = self.bn_layer(self.bn4(self.conv4(feature_cat)))

        return output.contiguous()


class DecoderBasicBlock(nn.Module):
    """Basic block for decoder ResNet architecture.
    Args:
        inplanes (int): Number of input channels.
        planes (int): Number of output channels.
        stride (int, optional): Stride for convolution and de-convolution layers. Defaults to 1.
        upsample (Optional[nn.Module], optional): Module used for upsampling output. Defaults to None.
        groups (int, optional): Number of blocked connections from input channels to output channels.
            Defaults to 1.
        base_width (int, optional): Number of layers in each intermediate convolution layer. Defaults to 64.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        norm_layer (Optional[Callable[..., nn.Module]], optional): Batch norm layer to use.Defaults to None.
    Raises:
        ValueError: If groups are not equal to 1 and base width is not 64.
        NotImplementedError: If dilation is greater than 1.
    """

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        upsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 2
        if stride == 2:
            self.conv1 = nn.ConvTranspose2d(
                inplanes, planes, kernel_size=2, stride=stride, groups=groups, bias=False, dilation=dilation
            )
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, batch: Tensor) -> Tensor:
        """Forward-pass of de-resnet block."""
        identity = batch

        out = self.conv1(batch)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(batch)

        out += identity
        out = self.relu(out)

        return out


class DecoderBottleneck(nn.Module):
    """Bottleneck for Decoder.
    Args:
        inplanes (int): Number of input channels.
        planes (int): Number of output channels.
        stride (int, optional): Stride for convolution and de-convolution layers. Defaults to 1.
        upsample (Optional[nn.Module], optional): Module used for upsampling output. Defaults to None.
        groups (int, optional): Number of blocked connections from input channels to output channels.
            Defaults to 1.
        base_width (int, optional): Number of layers in each intermediate convolution layer. Defaults to 64.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        norm_layer (Optional[Callable[..., nn.Module]], optional): Batch norm layer to use.Defaults to None.
    """

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        upsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 2
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        if stride == 2:
            self.conv2 = nn.ConvTranspose2d(
                width, width, kernel_size=2, stride=stride, groups=groups, bias=False, dilation=dilation
            )
        else:
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

    def forward(self, batch: Tensor) -> Tensor:
        """Forward-pass of de-resnet bottleneck block."""
        identity = batch

        out = self.conv1(batch)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            identity = self.upsample(batch)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """ResNet model for decoder.
    Args:
        block (Type[Union[DecoderBasicBlock, DecoderBottleneck]]): Type of block to use in a layer.
        layers (List[int]): List to specify number for blocks per layer.
        zero_init_residual (bool, optional): If true, initializes the last batch norm in each layer to zero.
            Defaults to False.
        groups (int, optional): Number of blocked connections per layer from input channels to output channels.
            Defaults to 1.
        width_per_group (int, optional): Number of layers in each intermediate convolution layer.. Defaults to 64.
        norm_layer (Optional[Callable[..., nn.Module]], optional): Batch norm layer to use. Defaults to None.
    """

    def __init__(
        self,
        block: Type[Union[DecoderBasicBlock, DecoderBottleneck]],
        layers: List[int],
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 512 * block.expansion
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.layer1 = self._make_layer(block, 256, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, DecoderBottleneck):
                    nn.init.constant_(module.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(module, DecoderBasicBlock):
                    nn.init.constant_(module.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[DecoderBasicBlock, DecoderBottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        upsample = None
        previous_dilation = self.dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=2,
                    stride=stride,
                    groups=self.groups,
                    bias=False,
                    dilation=self.dilation,
                ),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, upsample, self.groups, self.base_width, previous_dilation, norm_layer)
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, batch: Tensor) -> List[Tensor]:
        """Forward pass for Decoder ResNet. Returns list of features."""
        feature_a = self.layer1(batch)  # 512*8*8->256*16*16
        feature_b = self.layer2(feature_a)  # 256*16*16->128*32*32
        feature_c = self.layer3(feature_b)  # 128*32*32->64*64*64

        return [feature_c, feature_b, feature_a]


def _resnet(block: Type[Union[DecoderBasicBlock, DecoderBottleneck]], layers: List[int], **kwargs: Any) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    return model


def de_wide_resnet50_2() -> ResNet:
    """Wide ResNet-50-2 model."""
    return _resnet(DecoderBottleneck, [3, 4, 6, 3], width_per_group=128)


class AnomalyMapGenerator:
    """Generate Anomaly Heatmap.
    Args:
        image_size (Union[ListConfig, Tuple]): Size of original image used for upscaling the anomaly map.
        sigma (int): Standard deviation of the gaussian kernel used to smooth anomaly map.
        mode (str, optional): Operation used to generate anomaly map. Options are `add` and `multiply`.
                Defaults to "multiply".
    Raises:
        ValueError: In case modes other than multiply and add are passed.
    """

    def __init__(self, image_size: Tuple, sigma: int = 4, mode: str = "multiply"):
        self.image_size = image_size
        self.sigma = sigma
        self.kernel_size = 2 * int(4.0 * sigma + 0.5) + 1

        if mode not in ("add", "multiply"):
            raise ValueError(f"Found mode {mode}. Only multiply and add are supported.")
        self.mode = mode

    def __call__(self, student_features: List[Tensor], teacher_features: List[Tensor]) -> Tensor:
        """Computes anomaly map given encoder and decoder features.
        Args:
            student_features (List[Tensor]): List of encoder features
            teacher_features (List[Tensor]): List of decoder features
        Returns:
            Tensor: Anomaly maps of length batch.
        """
        if self.mode == "multiply":
            anomaly_map = torch.ones(
                [student_features[0].shape[0], 1, *self.image_size], device=student_features[0].device
            )  # b c h w
        elif self.mode == "add":
            anomaly_map = torch.zeros(
                [student_features[0].shape[0], 1, *self.image_size], device=student_features[0].device
            )

        for student_feature, teacher_feature in zip(student_features, teacher_features):
            distance_map = 1 - F.cosine_similarity(student_feature, teacher_feature)
            distance_map = torch.unsqueeze(distance_map, dim=1)
            distance_map = F.interpolate(distance_map, size=self.image_size, mode="bilinear", align_corners=True)
            if self.mode == "multiply":
                anomaly_map *= distance_map
            elif self.mode == "add":
                anomaly_map += distance_map

        anomaly_map = gaussian_blur2d(
            anomaly_map, kernel_size=(self.kernel_size, self.kernel_size), sigma=(self.sigma, self.sigma)
        )

        return anomaly_map


class ReverseDistillationModel(nn.Module):
    """Reverse Distillation Model.
    Args:
        encoder, oceb, decoder: model components previously declared
        input_size (Tuple[int, int]): Size of input image
        anomaly_map_mode (str): Mode used to generate anomaly map. Options are between ``multiply`` and ``add``.
    """

    def __init__(self, encoder, oceb, decoder, input_size=(224, 224), anomaly_map_mode: str = 'multiply'):
        super().__init__()

        for param in encoder.parameters():
            param.requires_grad = False
        self.encoder = encoder
        self.oceb = oceb
        self.decoder = decoder

        image_size = input_size

        self.anomaly_map_generator = AnomalyMapGenerator(image_size=tuple(image_size), mode=anomaly_map_mode)

    def forward(self, images: Tensor) -> Union[Tensor, Tuple[List[Tensor], List[Tensor]]]:
        """Forward-pass images to the network.
        During the training mode the model extracts features from encoder and decoder networks.
        During evaluation mode, it returns the predicted anomaly map.
        Args:
            images (Tensor): Batch of images
        Returns:
            Union[Tensor, Tuple[List[Tensor],List[Tensor]]]: Encoder and decoder features in training mode,
                else anomaly maps.
        """
        self.encoder.eval()

        encoder_features = self.encoder(images)
        #encoder_features = list(encoder_features.values())

        decoder_features = self.decoder(self.oceb(encoder_features))

        if self.training:
            output = encoder_features, decoder_features
        else:
            output = self.anomaly_map_generator(encoder_features, decoder_features)

        return output

