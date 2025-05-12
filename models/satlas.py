import satlaspretrain_models
import torch
from torch import nn
import collections
import torchvision
import requests
from io import BytesIO
from loguru import logger


import torch.nn
import torchvision

# Below function from https://github.com/allenai/satlaspretrain_models/blob/main/satlaspretrain_models/utils.py#L155
def adjust_state_dict_prefix(state_dict, needed, prefix=None, prefix_allowed_count=None):
    """
    Adjusts the keys in the state dictionary by replacing 'backbone.backbone' prefix with 'backbone'.

    Args:
        state_dict (dict): Original state dictionary with 'backbone.backbone' prefixes.

    Returns:
        dict: Modified state dictionary with corrected prefixes.
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        # Assure we're only keeping keys that we need for the current model component. 
        if not needed in key:
            continue

        # Update the key prefixes to match what the model expects.
        if prefix is not None:
            while key.count(prefix) > prefix_allowed_count:
                key = key.replace(prefix, '', 1)

        new_state_dict[key] = value
    return new_state_dict

def adjust_state_dict_prefix_modified(state_dict, needed, prefix=None, prefix_allowed_count=None):
    """
    Modified version for SwinB specifically
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        # Update the key prefixes to match what the model expects.
        if (needed in key) or (needed=="upsample"):
            if prefix is not None:
                while key.count(prefix) > prefix_allowed_count:
                    # logger.debug(f"prev key: {key}")
                    if needed=="fpn":
                        key = key.replace(prefix, '', 1)
                        key = "fpn."+key
                    elif needed=="upsample":
                        key = key.replace(prefix, 'upsample.',1)
                    # logger.debug(f"new key: {key}")

        new_state_dict[key] = value # keep original keys
    return new_state_dict

class SatlasModel(torch.nn.Module):
    def __init__(self, num_inp_feats=6, fpn=True, model_name="Sentinel2_SwinB_SI_RGB"):
        super(SatlasModel, self).__init__()

        weights_manager = satlaspretrain_models.Weights()
        if num_inp_feats != 3:
            self.first = nn.Conv2d(num_inp_feats, 3, 1) # from 6 channels to 3
        else:
            self.first = nn.Identity()
        if model_name == "Sentinel2_SwinB_SI_RGB":
            try:
                logger.debug("Loading Satlas SwinB weights")
                self.backbone = weights_manager.get_pretrained_model(model_identifier=model_name, fpn=fpn)
                self.backbone_channels = self.backbone.upsample.layers[-1][-2].out_channels
            except Exception as e:
                logger.warning(f"Error loading from weights_manager: {e}")
                model = ProxySatlas(model_identifier="Sentinel2_SwinB_SI_RGB")
                # logger.debug(f"self.backbone: {self.backbone}")
                weights = torch.load("checkpoints/satlas/sentinel2_swinb_si_rgb.pth")
                fpn_state_dict = adjust_state_dict_prefix_modified(weights, 'fpn', 'intermediates.0.', 0)
                fpn_state_dict = adjust_state_dict_prefix_modified(fpn_state_dict, 'upsample', 'intermediates.1.', 0)
                tmp = model.load_state_dict(fpn_state_dict, strict=False)
                logger.debug(f"tmp: {tmp.missing_keys}")
                self.backbone = model
                self.backbone_channels = 128
        elif model_name == "Sentinel2_SwinT_SI_RGB":
            logger.debug("Loading Satlas SwinT weights")
            try:
                model = weights_manager.get_pretrained_model(model_identifier=model_name, fpn=False)
            except Exception as e: 
                logger.warning(f"weights_manager failed: {e}")
                model = ProxySatlas(model_identifier="Sentinel2_SwinT_SI_RGB")
                local_file = "checkpoints/satlas/sentinel2_swint_si_rgb.pth"
                local_weights = torch.load(local_file)
                tmp = model.load_state_dict(local_weights, strict=False)
                logger.debug(f"Missing keys from loaded weights: {tmp.missing_keys}")
            out_channels = [
                [4, 96],
                [8, 192],
                [16, 384],
                [32, 768],
            ]
            model_fpn = FPN(out_channels, 128)
            if fpn: # Download and load weights for FPN
                weights_url = 'https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_swint_si_rgb.pth?download=true'
                response = requests.get(weights_url)
                if response.status_code == 200:
                    weights_file = BytesIO(response.content)
                else:
                    logger.warning(f"Using local file. Error downloading fpn file : {weights_url}")
                    weights_file = "checkpoints/satlas/sentinel2_swint_si_rgb.pth"
                weights = torch.load(weights_file)
                fpn_state_dict = adjust_state_dict_prefix(weights, 'fpn', 'intermediates.0.', 0)
                model_fpn.load_state_dict(fpn_state_dict, strict=True)
            model_upsample = Upsample(model_fpn.out_channels)
            self.backbone = torch.nn.Sequential(
                model,
                model_fpn,
                model_upsample,
            )
            self.backbone_channels = 128
        elif model_name == "Sentinel2_Resnet50_SI_RGB":
            logger.debug("Loading Satlas RN50 weights")
            try:
                model = weights_manager.get_pretrained_model(model_identifier="Sentinel2_Resnet50_SI_RGB", fpn=False)
                model.backbone.freeze_bn = False    # NOTE: means backbone is not frozen during training
            except Exception as e: 
                logger.warning(f"weights_manager failed: {e}")
                local_file = "checkpoints/satlas/sentinel2_resnet50_si_rgb.pth"
                model = ProxySatlas(model_identifier="Sentinel2_Resnet50_SI_RGB")
                local_weights = torch.load(local_file)
                tmp = model.load_state_dict(local_weights, strict=False)
                logger.debug(f"Missing keys from loaded weights: {tmp.missing_keys}")
            out_channels = [
                [4, 256],
                [8, 512],
                [16, 1024],
                [32, 2048],
            ]
            model_fpn = FPN(out_channels, 128)
            if fpn:
                weights_url = 'https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_resnet50_si_rgb.pth?download=true'
                response = requests.get(weights_url)
                logger.debug(f"response: {response}")
                if response.status_code == 200:
                    weights_file = BytesIO(response.content)
                else:
                    logger.warning(f"Using local file. Error downloading fpn file : {weights_url}")
                    weights_file = "checkpoints/satlas/sentinel2_resnet50_si_rgb.pth"
                weights = torch.load(weights_file)
                fpn_state_dict = adjust_state_dict_prefix(weights, 'fpn', 'intermediates.0.', 0)
                model_fpn.load_state_dict(fpn_state_dict, strict=True)
            model_upsample = Upsample(model_fpn.out_channels)
            self.backbone = torch.nn.Sequential(
                model,
                model_fpn,
                model_upsample,
            )
            self.backbone_channels = 128

    def forward(self, x):
        x = self.first(x)
        x = self.backbone(x)
        return x[0]

# Below classes from https://github.com/allenai/satlaspretrain_models/blob/main/satlaspretrain_models/models
class SatlasHead(torch.nn.Module):
    def __init__(self, backbone_channels, out_channels):
        super(SatlasHead, self).__init__()

        num_layers = 2
        layers = []
        for _ in range(num_layers-1):
            layer = torch.nn.Sequential(
                torch.nn.Conv2d(backbone_channels, backbone_channels, 3, padding=1),
                torch.nn.ReLU(inplace=True),
            )
            layers.append(layer)
        layers.append(torch.nn.Conv2d(backbone_channels, out_channels, 3, padding=1))
        self.head = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.head(x)


# Below classes from https://github.com/allenai/satlaspretrain_models/blob/main/satlaspretrain_models/models/fpn.py#L6
class FPN(torch.nn.Module):
    def __init__(self, backbone_channels, out_channels):    # NOTE: modified out_channels to match checkpoint
        super(FPN, self).__init__()

        # out_channels = backbone_channels[0][1]
        in_channels_list = [ch[1] for ch in backbone_channels]
        self.fpn = torchvision.ops.FeaturePyramidNetwork(in_channels_list=in_channels_list, out_channels=out_channels)

        self.out_channels = [[ch[0], out_channels] for ch in backbone_channels]

    def forward(self, x):
        inp = collections.OrderedDict([('feat{}'.format(i), el) for i, el in enumerate(x)])
        output = self.fpn(inp)
        output = list(output.values())

        return output


class Upsample(torch.nn.Module):
    # Computes an output feature map at 1x the input resolution.
    # It just applies a series of transpose convolution layers on the
    # highest resolution features from the backbone (FPN should be applied first).

    def __init__(self, backbone_channels):
        super(Upsample, self).__init__()
        self.in_channels = backbone_channels

        out_channels = backbone_channels[0][1]
        self.out_channels = [(1, out_channels)] + backbone_channels

        layers = []
        depth, ch = backbone_channels[0]
        while depth > 1:
            next_ch = max(ch//2, out_channels)
            layer = torch.nn.Sequential(
                torch.nn.Conv2d(ch, ch, 3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.ConvTranspose2d(ch, next_ch, 4, stride=2, padding=1),
                torch.nn.ReLU(inplace=True),
            )
            layers.append(layer)
            ch = next_ch
            depth /= 2

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        output = self.layers(x[0])
        return [output] + x



class ProxySatlas(torch.nn.Module):
    def __init__(self, model_identifier):
        super(ProxySatlas, self).__init__()
        self.model_identifier = model_identifier
        if model_identifier=="Sentinel2_Resnet50_SI_RGB":
            backbone = ResnetBackbone(num_channels=3)
        elif model_identifier=="Sentinel2_SwinB_SI_RGB":
            backbone = SwinBackbone(num_channels=3, arch="swinb")
            out_channels = [
                [4, 128],
                [8, 256],
                [16, 512],
                [32, 1024],
            ]
            self.fpn = FPN(out_channels, 128)
            # if fpn: # Download and load weights for FPN
            #     weights_url = 'https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_swint_si_rgb.pth?download=true'
            #     response = requests.get(weights_url)
            #     if response.status_code == 200:
            #         weights_file = BytesIO(response.content)
            #     weights = torch.load(weights_file)
            #     fpn_state_dict = adjust_state_dict_prefix(weights, 'fpn', 'intermediates.0.', 0)
            #     fpn.load_state_dict(fpn_state_dict)
            self.upsample = Upsample(self.fpn.out_channels)
        elif model_identifier=="Sentinel2_SwinT_SI_RGB":
            backbone = SwinBackbone(num_channels=3, arch="swint")
        self.backbone = backbone
    
    def forward(self, x):
        if self.model_identifier == "Sentinel2_SwinB_SI_RGB":
            x = self.backbone(x)
            x = self.fpn(x)
            x = self.upsample(x)
            return x
        else:
            return self.backbone(x)


# Below are from https://github.com/allenai/satlaspretrain_models/blob/main/satlaspretrain_models/models/backbones.py#L37




class SwinBackbone(torch.nn.Module):
    def __init__(self, num_channels, arch):
        super(SwinBackbone, self).__init__()

        if arch == 'swinb':
            self.backbone = torchvision.models.swin_v2_b()
            self.out_channels = [
                [4, 128],
                [8, 256],
                [16, 512],
                [32, 1024],
            ]
        elif arch == 'swint':
            self.backbone = torchvision.models.swin_v2_t()
            self.out_channels = [
                [4, 96],
                [8, 192],
                [16, 384],
                [32, 768],
            ]
        else:
            raise ValueError("Backbone architecture not supported.")

        self.backbone.features[0][0] = torch.nn.Conv2d(num_channels, self.backbone.features[0][0].out_channels, kernel_size=(4, 4), stride=(4, 4))

    def forward(self, x):
        outputs = []
        for layer in self.backbone.features:
            x = layer(x)
            outputs.append(x.permute(0, 3, 1, 2))
        return [outputs[-7], outputs[-5], outputs[-3], outputs[-1]]


class ResnetBackbone(torch.nn.Module):
    def __init__(self, num_channels, arch='resnet50'):
        super(ResnetBackbone, self).__init__()

        if arch == 'resnet50':
            self.resnet = torchvision.models.resnet.resnet50(weights=None)
            ch = [256, 512, 1024, 2048]
        elif arch == 'resnet152':
            self.resnet = torchvision.models.resnet.resnet152(weights=None)
            ch = [256, 512, 1024, 2048]
        else:
            raise ValueError("Backbone architecture not supported.")

        self.resnet.conv1 = torch.nn.Conv2d(num_channels, self.resnet.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.out_channels = [
            [4, ch[0]],
            [8, ch[1]],
            [16, ch[2]],
            [32, ch[3]],
        ]

    def train(self, mode=True):
        super(ResnetBackbone, self).train(mode)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        layer1 = self.resnet.layer1(x)
        layer2 = self.resnet.layer2(layer1)
        layer3 = self.resnet.layer3(layer2)
        layer4 = self.resnet.layer4(layer3)

        return [layer1, layer2, layer3, layer4]


class AggregationBackbone(torch.nn.Module):
    def __init__(self, num_channels, backbone):
        super(AggregationBackbone, self).__init__()

        # Number of channels to pass to underlying backbone.
        self.image_channels = num_channels

        # Prepare underlying backbone.
        self.backbone = backbone

        # Features from images within each group are aggregated separately.
        # Then the output is the concatenation across groups.
        # e.g. [[0], [1, 2]] to compare first image against the others
        self.groups = [[0, 1, 2, 3, 4, 5, 6, 7]]

        ngroups = len(self.groups)
        self.out_channels = [(depth, ngroups*count) for (depth, count) in self.backbone.out_channels]

        self.aggregation_op = 'max'

    def forward(self, x):
        # First get features of each image.
        all_features = []
        for i in range(0, x.shape[1], self.image_channels):
            features = self.backbone(x[:, i:i+self.image_channels, :, :])
            all_features.append(features)

        # Now compute aggregation over each group.
        # We handle each depth separately.
        l = []
        for feature_idx in range(len(all_features[0])):
            aggregated_features = []
            for group in self.groups:
                group_features = []
                for image_idx in group:
                    # We may input fewer than the maximum number of images.
                    # So here we skip image indices in the group that aren't available.
                    if image_idx >= len(all_features):
                        continue

                    group_features.append(all_features[image_idx][feature_idx])
                # Resulting group features are (depth, batch, C, height, width).
                group_features = torch.stack(group_features, dim=0)

                if self.aggregation_op == 'max':
                    group_features = torch.amax(group_features, dim=0)

                aggregated_features.append(group_features)

            # Finally we concatenate across groups.
            aggregated_features = torch.cat(aggregated_features, dim=1)

            l.append(aggregated_features)

        return l