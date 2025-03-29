import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
import models.PointBind_models as models
from imagebind.imagebind_model import ModalityType
import numpy as np
import logging

# Mapping from modality string -> the attribute name for that MLP
MODALITY_TO_MLP = {
    "image":   "mlp_for_image",
    "video":   "mlp_for_video",
    "audio":   "mlp_for_audio",
    "thermal": "mlp_for_thermal",
    "point":   "mlp_for_point",
    "event":   "mlp_for_event",
}

class UniBind(nn.Module):
    def __init__(self, args, use_flash_attention = False, fine_tuned_weights=None, logger=None):
        super(UniBind, self).__init__()
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

        self.modality = args.modality
        self.backbone = models.PointBind_I2PMAE(use_flash_attention=use_flash_attention)

        state_dict = torch.load(args.pretrain_weights, weights_only=True, map_location='cpu')
        self.backbone.load_state_dict(state_dict, strict=True)
        for param in self.backbone.parameters():
            param.requires_grad_(False)

        if self.modality == "image":
            self.mlp_for_image = init_linear_as_identity(nn.Linear(1024, 1024))
        elif self.modality == "video":
            self.mlp_for_video = init_linear_as_identity(nn.Linear(1024, 1024))
        elif self.modality == "audio":
            self.mlp_for_audio = init_linear_as_identity(nn.Linear(1024, 1024))
        elif self.modality == "thermal":
            self.mlp_for_thermal = init_linear_as_identity(nn.Linear(1024, 1024))
        elif self.modality == "point":
            self.mlp_for_point = init_linear_as_identity(nn.Linear(1024, 1024))
        elif self.modality == "event":
            self.mlp_for_event = init_linear_as_identity(nn.Linear(1024, 1024))
        else:
            raise ValueError(f"Unsupported modality: {self.modality}")

        # ---------------------------------------------------------------------
        # If the user provided a path with fine-tuned MLP weights, load them
        # immediately in the constructor:
        # ---------------------------------------------------------------------
        if fine_tuned_weights is not None:
            self.logger.info(f"[UniBind init] Loading MLP submodules from '{fine_tuned_weights}'...")
            self.load_fine_tuned_weights(fine_tuned_weights)

    def forward(self, inputs):
        if self.modality == "image":
            outputs = self.__bind(inputs)
            text_embeddings = outputs[ModalityType.TEXT]
            vision_embeddings = self.mlp_for_image(outputs[ModalityType.VISION])
        if self.modality == "video":
            outputs = self.__bind(inputs)
            text_embeddings = outputs[ModalityType.TEXT]
            vision_embeddings = self.mlp_for_video(outputs[ModalityType.VISION])
        if self.modality == "audio":
            outputs = self.__bind(inputs)
            text_embeddings = outputs[ModalityType.TEXT]
            vision_embeddings = self.mlp_for_audio(outputs[ModalityType.AUDIO])
        if self.modality == "thermal":
            outputs = self.__bind(inputs)
            text_embeddings = outputs[ModalityType.TEXT]
            vision_embeddings = self.mlp_for_thermal(outputs[ModalityType.THERMAL])
        if self.modality == "event":
            outputs = self.__bind(inputs)
            text_embeddings = outputs[ModalityType.TEXT]
            vision_embeddings = self.mlp_for_event(outputs[ModalityType.VISION])
        if self.modality == "point":
            pc_embeddings = self.backbone.encode_pc(inputs['point'])
            pc_features = self.backbone.bind.modality_head_point(pc_features)
            pc_features = self.backbone.bind.modality_postprocessor_point(pc_features)
            outputs = self.__bind({ModalityType.TEXT: inputs['text']})
            text_embeddings = outputs[ModalityType.TEXT]
            vision_embeddings = self.mlp_for_point(pc_embeddings)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        vision_embeddings = vision_embeddings / vision_embeddings.norm(dim=-1, keepdim=True)
        return text_embeddings, vision_embeddings
    
    def encode_vision(self, inputs):
        if self.modality == "image":
            outputs = self.backbone.bind(inputs)
            vision_embeddings = outputs[ModalityType.VISION]
        if self.modality == "video":
            outputs = self.backbone.bind(inputs)
            vision_embeddings = outputs[ModalityType.VISION]
        if self.modality == "audio":
            outputs = self.backbone.bind(inputs)
            vision_embeddings = outputs[ModalityType.AUDIO]
        if self.modality == "thermal":
            outputs = self.backbone.bind(inputs)
            vision_embeddings = outputs[ModalityType.THERMAL]
        if self.modality == "event":
            outputs = self.backbone.bind(inputs)
            vision_embeddings = outputs[ModalityType.VISION]
        if self.modality == "point":
            pc_embeddings = self.backbone.encode_pc(inputs['point'])
            pc_embeddings = self.backbone.bind.modality_head_point(pc_embeddings)
            vision_embeddings = self.backbone.bind.modality_postprocessor_point(pc_embeddings)
        vision_embeddings = vision_embeddings / vision_embeddings.norm(dim=-1, keepdim=True)
        return vision_embeddings
    
    def encode_vision_with_mlp(self, inputs):
        if self.modality == "image":
            outputs = self.backbone.bind(inputs)
            vision_embeddings = self.mlp_for_image(outputs[ModalityType.VISION])
        if self.modality == "video":
            outputs = self.backbone.bind(inputs)
            vision_embeddings = self.mlp_for_video(outputs[ModalityType.VISION])
        if self.modality == "audio":
            outputs = self.backbone.bind(inputs)
            vision_embeddings = self.mlp_for_audio(outputs[ModalityType.AUDIO])
        if self.modality == "thermal":
            outputs = self.backbone.bind(inputs)
            vision_embeddings = self.mlp_for_thermal(outputs[ModalityType.THERMAL])
        if self.modality == "event":
            outputs = self.backbone.bind(inputs)
            vision_embeddings = self.mlp_for_event(outputs[ModalityType.VISION])
        if self.modality == "point":
            pc_embeddings = self.backbone.encode_pc(inputs['point'])
            pc_embeddings = self.backbone.modality_head_point(pc_embeddings)
            pc_embeddings = self.backbone.modality_postprocessor_point(pc_embeddings)
            vision_embeddings = self.mlp_for_point(pc_embeddings)
        vision_embeddings = vision_embeddings / vision_embeddings.norm(dim=-1, keepdim=True)
        return vision_embeddings
    

    def encode_text(self, inputs):
        text_embeddings = self.backbone.bind(inputs)[ModalityType.TEXT]
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        return text_embeddings
    
    # -------------------------------------------------------------------------
    # Save / Load methods for ALL MLP submodules in this model
    # -------------------------------------------------------------------------
    def save_fine_tuned_weights(self, checkpoint_path: str):
        """
        Saves all MLP submodules (if they exist) into a single .pt file,
        as a dict keyed by the submodule attribute name (e.g. "mlp_for_image").
        """
        mlps_dict = {}
        for mod, mlp_attr in MODALITY_TO_MLP.items():
            if hasattr(self, mlp_attr):
                mlp_module = getattr(self, mlp_attr)
                mlps_dict[mlp_attr] = mlp_module.state_dict()

        torch.save(mlps_dict, checkpoint_path)
        self.logger.info(f"[save_modality_mlps] Saved all MLP submodules to '{checkpoint_path}'.")

    def load_fine_tuned_weights(self, checkpoint_path: str, map_location='cpu'):
        """
        Loads MLP submodules from a single .pt file (created by save_modality_mlps()).
        Only updates MLPs that already exist on this object. 
        """
        mlps_dict = torch.load(checkpoint_path, map_location=map_location)

        for mlp_attr, state_dict in mlps_dict.items():
            # If this model actually has that submodule, load it:
            if hasattr(self, mlp_attr):
                getattr(self, mlp_attr).load_state_dict(state_dict)
                self.logger.info(f"[load_fine_tuned_weights] Loaded '{mlp_attr}' from '{checkpoint_path}'.")
            else:
                self.logger.warning(f"[load_fine_tuned_weights] This model has no '{mlp_attr}' attribute. Skipping.")
    
    def __bind(self, inputs):
        with torch.inference_mode():
            return self.backbone.bind(inputs)
    
def init_linear_as_identity(linear_layer):
    assert linear_layer.in_features == linear_layer.out_features, \
        "Cannot set identity: layer must be square."
    # Initialize weight to identity
    init.eye_(linear_layer.weight)
    # Initialize bias to zeros
    nn.init.zeros_(linear_layer.bias)
    return linear_layer