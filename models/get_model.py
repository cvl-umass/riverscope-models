#
# Authors: Wei-Hong Li

import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.data import DataLoader

from models.satlas import SatlasModel, SatlasHead
import segmentation_models_pytorch as smp
from loguru import logger

def get_model(args, tasks_outputs, num_inp_feats=3, pretrained=True):
    backbone_channels = None
    segmodel_in_channels = 3
    if args.adaptor == "no_init":
        segmodel_in_channels = num_inp_feats
        logger.warning(f"Using no_init as adaptor. setting pretrained to False.")
        pretrained = False
        logger.warning(f"pretrained={pretrained}")
    if args.adaptor == "drop":
        logger.warning(f"[DROP] Changing num_inp_feats to 3 since dropping non-RGB band")
        num_inp_feats = 3
    logger.debug(f"num_inp_feats: {num_inp_feats}")
    if args.segment_model in ["deeplabv3", "deeplabv3plus", "unet", "fpn"]:
        weights = "imagenet" if pretrained else None
        logger.debug(f"Using encoder weights: {weights}")
        assert args.backbone in [
            "resnet50", "mobilenet_v2", "resnet50_mocov3", "resnet50_seco", 
            "swint", "swinb",
            "satlas_si_swinb", "satlas_si_swint", "satlas_si_resnet50"
        ]
        if args.backbone == "resnet50": # imagenet
            rn50_segmodel = smp.create_model(
                arch=args.segment_model,                     # name of the architecture, e.g. 'Unet'/ 'FPN' / etc. Case INsensitive!
                encoder_name="resnet50",
                encoder_weights=weights,
                in_channels=segmodel_in_channels,  # NOTE: always 3, change number of input channels in ModelwithAdaptor
                classes=1,
            )
            backbone = ModelwithAdaptor(adaptor=args.adaptor, backbone=rn50_segmodel, num_inp_feats=num_inp_feats)
            args.head = "no_head"
            logger.warning(f"There should be no additional head. args.head: {args.head}")
        elif args.backbone == "resnet50_mocov3":
            rn50_segmodel = smp.create_model(
                arch=args.segment_model,                     # name of the architecture, e.g. 'Unet'/ 'FPN' / etc. Case INsensitive!
                encoder_name="resnet50",
                encoder_weights="imagenet",
                in_channels=segmodel_in_channels,  # NOTE: always 3, change number of input channels in ModelwithAdaptor
                classes=1,
            )
            # Using MoCo pre-trained deeplabv3+ model
            state_dict = torch.load("checkpoints/moco_v3/r-50-100ep.pth.tar")['state_dict']
            new_state_dict = {}
            linear_keyword = 'fc'
            for k in list(state_dict.keys()):    # from MoCo (https://github.com/facebookresearch/moco-v3/blob/c349e6e24f40d3fedb22d973f92defa4cedf37a7/main_lincls.py#L179)
                # retain only base_encoder up to before the embedding layer
                if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
                    # remove prefix
                    new_state_dict[k[len("module.base_encoder."):]] = state_dict[k]
            rn50_segmodel.encoder.load_state_dict(new_state_dict, strict=True)
            logger.debug(f"Loaded checkpoints/moco_v3/r-50-100ep.pth.tar")
            backbone = ModelwithAdaptor(adaptor=args.adaptor, backbone=rn50_segmodel, num_inp_feats=num_inp_feats)
            args.head = "no_head"
            logger.warning(f"There should be no additional head. args.head: {args.head}")
        elif args.backbone == "resnet50_seco":
            rn50_segmodel = smp.create_model(
                arch=args.segment_model,                     # name of the architecture, e.g. 'Unet'/ 'FPN' / etc. Case INsensitive!
                encoder_name="resnet50",
                encoder_weights="imagenet",
                in_channels=segmodel_in_channels,  # NOTE: always 3, change number of input channels in ModelwithAdaptor
                classes=1,
            )
            from models.seco import MocoV2
            from collections import OrderedDict

            seco_model = MocoV2.load_from_checkpoint("checkpoints/seco/seco_resnet50_1m.ckpt")
            encoder = seco_model.encoder_q

            new_names = ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool", "flatten"]
            enc = torch.nn.Sequential(OrderedDict(zip(new_names, encoder.children())))
            rn50_segmodel.encoder.load_state_dict(enc.state_dict(), strict=True)
            logger.debug(f"Loaded checkpoints/seco/seco_resnet50_1m.ckpt")
            backbone = ModelwithAdaptor(adaptor=args.adaptor, backbone=rn50_segmodel, num_inp_feats=num_inp_feats)
            args.head = "no_head"
            logger.warning(f"There should be no additional head. args.head: {args.head}")
        elif args.backbone == "mobilenet_v2":
            mobilenetv2_segmodel = smp.create_model(
                arch=args.segment_model,                     # name of the architecture, e.g. 'Unet'/ 'FPN' / etc. Case INsensitive!
                encoder_name="mobilenet_v2",
                encoder_weights=weights,
                in_channels=segmodel_in_channels,
                classes=1,
            )
            logger.debug(f"Using mobilenetv2+")
            backbone = ModelwithAdaptor(adaptor=args.adaptor, backbone=mobilenetv2_segmodel, num_inp_feats=num_inp_feats)
            args.head = "no_head"
            logger.warning(f"There should be no additional head. args.head: {args.head}")
        elif args.backbone == "swint":
            swint_segmodel = smp.create_model(
                arch=args.segment_model,                     # name of the architecture, e.g. 'Unet'/ 'FPN' / etc. Case INsensitive!
                encoder_name="tu-swin_s3_tiny_224.ms_in1k",
                encoder_weights=weights,
                in_channels=segmodel_in_channels,
                classes=1,
            )
            logger.debug(f"Using swint")
            backbone = ModelwithAdaptor(adaptor=args.adaptor, backbone=swint_segmodel, num_inp_feats=num_inp_feats)
            args.head = "no_head"
            logger.warning(f"There should be no additional head. args.head: {args.head}")
        elif args.backbone == "swinb":
            swinb_segmodel = smp.create_model(
                arch=args.segment_model,                     # name of the architecture, e.g. 'Unet'/ 'FPN' / etc. Case INsensitive!
                encoder_name="tu-swin_s3_base_224.ms_in1k",
                encoder_weights=weights,
                in_channels=segmodel_in_channels,
                classes=1,
            )
            logger.debug(f"Using swinb")
            backbone = ModelwithAdaptor(adaptor=args.adaptor, backbone=swinb_segmodel, num_inp_feats=num_inp_feats)
            args.head = "no_head"
            logger.warning(f"There should be no additional head. args.head: {args.head}")
        elif args.backbone == "satlas_si_swinb":
            if not pretrained:
                raise NotImplementedError
            if args.segment_model in ["deeplabv3plus", "unet"]:
                raise NotImplementedError
            backbone = SatlasModel(num_inp_feats=num_inp_feats, model_name="Sentinel2_SwinB_SI_RGB")
            backbone_channels = backbone.backbone_channels
        elif args.backbone == "satlas_mi_swinb":
            if not pretrained:
                raise NotImplementedError
            if args.segment_model in ["deeplabv3plus", "unet"]:
                raise NotImplementedError
            backbone = SatlasModel(num_inp_feats=num_inp_feats, model_name="Sentinel2_SwinB_MI_RGB")
            backbone_channels = backbone.backbone_channels
        elif args.backbone == "satlas_si_swint":    # NOTE: non-swinB models have a bug (made a fix)
            if not pretrained:
                raise NotImplementedError
            if args.segment_model in ["deeplabv3plus", "unet"]:
                raise NotImplementedError
            backbone = SatlasModel(num_inp_feats=num_inp_feats, model_name="Sentinel2_SwinT_SI_RGB")    
            backbone_channels = backbone.backbone_channels
        elif args.backbone == "satlas_si_resnet50":    # NOTE: non-swinB models have a bug (made a fix)
            if not pretrained:
                raise NotImplementedError
            if args.segment_model in ["deeplabv3plus", "unet"]:
                raise NotImplementedError
            backbone = SatlasModel(num_inp_feats=num_inp_feats, model_name="Sentinel2_Resnet50_SI_RGB")
            backbone_channels = backbone.backbone_channels
    elif args.segment_model == "dpt":
        weights = "imagenet" if pretrained else None    # NOTE: None for random weights
        logger.debug(f"Using encoder weights: {weights} [None for random weights]")
        assert args.backbone in [
            "vitb", "vitb_dino", "vitb_mocov3", "vitb_clip", "vitb_prithvi",
            "vitl", 
            
        ]
        if args.backbone == "vitb":
            vitb16_dpt = smp.create_model(
                arch="dpt",                     # name of the architecture, e.g. 'Unet'/ 'FPN' / etc. Case INsensitive!
                encoder_name="tu-vit_base_patch16_224.orig_in21k",
                encoder_weights=weights,
                in_channels=segmodel_in_channels,
                classes=1,
            )
            backbone = ModelwithAdaptor(adaptor=args.adaptor, backbone=vitb16_dpt, num_inp_feats=num_inp_feats)
            logger.debug("Using tu-vit_base_patch16_224.orig_in21k")
        elif args.backbone == "vitl":
            vitb16_dpt = smp.create_model(
                arch="dpt",                     # name of the architecture, e.g. 'Unet'/ 'FPN' / etc. Case INsensitive!
                encoder_name="tu-vit_large_patch16_224.orig_in21k",
                encoder_weights=weights,
                in_channels=segmodel_in_channels,
                classes=1,
            )
            backbone = ModelwithAdaptor(adaptor=args.adaptor, backbone=vitb16_dpt, num_inp_feats=num_inp_feats)
            logger.debug("Using tu-vit_large_patch16_224.orig_in21k")
        elif args.backbone == "vitb_dino":
            vitb16_dpt = smp.create_model(
                arch="dpt",                     # name of the architecture, e.g. 'Unet'/ 'FPN' / etc. Case INsensitive!
                encoder_name="tu-vit_base_patch16_224.dino",
                encoder_weights=weights,
                in_channels=segmodel_in_channels,
                classes=1,
            )
            backbone = ModelwithAdaptor(adaptor=args.adaptor, backbone=vitb16_dpt, num_inp_feats=num_inp_feats)
            logger.debug("Using tu-vit_base_patch16_224.dino")
        elif args.backbone == "vitb_clip":
            vitb16_dpt = smp.create_model(
                arch="dpt",                     # name of the architecture, e.g. 'Unet'/ 'FPN' / etc. Case INsensitive!
                encoder_name="tu-vit_base_patch16_clip_224.laion2b",
                encoder_weights=weights,
                in_channels=segmodel_in_channels,
                classes=1,
            )
            backbone = ModelwithAdaptor(adaptor=args.adaptor, backbone=vitb16_dpt, num_inp_feats=num_inp_feats)
            logger.debug("Using tu-vit_base_patch16_clip_224.laion2b")
        elif args.backbone == "vitb_mocov3":
            vitb16_dpt = smp.create_model(
                arch="dpt",                     # name of the architecture, e.g. 'Unet'/ 'FPN' / etc. Case INsensitive!
                encoder_name="tu-vit_base_patch16_224.dino",
                encoder_weights=weights,
                in_channels=segmodel_in_channels,
                classes=1,
            )
            state_dict = torch.load("checkpoints/moco_v3/vit-b-300ep.pth.tar")['state_dict']
            new_state_dict = {}
            linear_keyword = 'head'
            for k in list(state_dict.keys()):	# from MoCo (https://github.com/facebookresearch/moco-v3/blob/c349e6e24f40d3fedb22d973f92defa4cedf37a7/main_lincls.py#L179)
                # retain only base_encoder up to before the embedding layer
                if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
                    # remove prefix
                    new_state_dict[k[len("module.base_encoder."):]] = state_dict[k]
            msg = vitb16_dpt.encoder.model.load_state_dict(new_state_dict, strict=True)
            backbone = ModelwithAdaptor(adaptor=args.adaptor, backbone=vitb16_dpt, num_inp_feats=num_inp_feats)
            logger.debug(f"Loading checkpoint for mocov3 vitb16: {msg}")
        elif args.backbone == "vitb_prithvi":
            if args.adaptor in ["drop", "no_init"]:
                raise NotImplementedError
            vitb16_dpt_6chan = smp.create_model(
                arch="dpt",                     # name of the architecture, e.g. 'Unet'/ 'FPN' / etc. Case INsensitive!
                encoder_name="tu-vit_base_patch16_224.orig_in21k",
                encoder_weights="imagenet",
                in_channels=6,  # NOTE: prithvi accepts 6 channels by default
                classes=1,
            )
            weights_path = "checkpoints/prithvi/Prithvi_100M.pt"
            prithvi_checkpoint = torch.load(weights_path, map_location="cpu")
            del prithvi_checkpoint['pos_embed'] # pos_embed is different
            del prithvi_checkpoint['decoder_pos_embed']
            prithvi_checkpoint["patch_embed.proj.weight"] = prithvi_checkpoint["patch_embed.proj.weight"].squeeze(2)    # remove time dimension

            msg = vitb16_dpt_6chan.encoder.model.load_state_dict(prithvi_checkpoint, strict=False)
            logger.debug(f"Loading checkpoint for prithvi vitb16: missing_keys={msg.missing_keys}")
            backbone = ModelwithAdaptor(adaptor=args.adaptor, backbone=vitb16_dpt_6chan, num_inp_feats=num_inp_feats, out_channels=6)
        args.head = "no_head"
        logger.warning(f"There should be no additional head. args.head: {args.head}")
    else:   # default is the old model setup
        if args.backbone == "satlas_si_swinb":
            if not pretrained:
                raise NotImplementedError
            backbone = SatlasModel(num_inp_feats=num_inp_feats, model_name="Sentinel2_SwinB_SI_RGB")
            backbone_channels = backbone.backbone_channels
        elif args.backbone == "satlas_mi_swinb":
            if not pretrained:
                raise NotImplementedError
            backbone = SatlasModel(num_inp_feats=num_inp_feats, model_name="Sentinel2_SwinB_MI_RGB")
            backbone_channels = backbone.backbone_channels
        elif args.backbone == "satlas_si_swint":    # NOTE: non-swinB models have a bug (made a fix)
            if not pretrained:
                raise NotImplementedError
            backbone = SatlasModel(num_inp_feats=num_inp_feats, model_name="Sentinel2_SwinT_SI_RGB")    
            backbone_channels = backbone.backbone_channels
        elif args.backbone == "satlas_si_resnet50":    # NOTE: non-swinB models have a bug (made a fix)
            if not pretrained:
                raise NotImplementedError
            backbone = SatlasModel(num_inp_feats=num_inp_feats, model_name="Sentinel2_Resnet50_SI_RGB")
            backbone_channels = backbone.backbone_channels

    if args.method == 'single-task':
        from models.models import SingleTaskModel
        task = args.task
        head = get_head(args.head, backbone_channels, tasks_outputs[task])
        model = SingleTaskModel(backbone, head, task)
    elif args.method == 'vanilla':
        selected_tasks_outputs = {}
        for task, task_output in tasks_outputs.items():
            if task in args.tasks:
                selected_tasks_outputs[task] = task_output
        from models.models import MultiTaskModel
        logger.debug(f"backbone_channels: {backbone_channels}")
        heads = torch.nn.ModuleDict({task: get_head(args.head, backbone_channels, task_output) for task, task_output in zip(args.tasks, selected_tasks_outputs.values())})
        model = MultiTaskModel(backbone, heads, args.tasks)

    return model


def get_head(head, backbone_channels, task_output):
    """ Return the decoder head """
    if head == "satlas_head":
        return SatlasHead(backbone_channels=backbone_channels, out_channels=task_output)
    elif (head == "unet_head") or (head == "no_head"):
        return nn.Identity()    # NOTE: Identity because decoder is incorporated into the backbone

class ModelwithAdaptor(torch.nn.Module):
    def __init__(self, adaptor, backbone, num_inp_feats=4, out_channels=3): #resnetv2_50
        super(ModelwithAdaptor, self).__init__()
        assert adaptor in ["linear", "drop", "no_init"]
        if adaptor=="linear" and ((num_inp_feats != 3) or (out_channels != num_inp_feats)):
            logger.debug(f"Using Conv2D adaptor")
            self.adaptor = nn.Conv2d(num_inp_feats, out_channels, 1) # from num_inp_feats channels to 3 (out_channels)
        else:
            logger.debug(f"Using identity adaptor -- effectively no adaptor")
            self.adaptor = nn.Identity()    # if random init, no need for adaptor. if drop, also no need
        self.backbone = backbone
    def forward(self, x):
        out = self.adaptor(x)
        out = self.backbone(out)
        return out

