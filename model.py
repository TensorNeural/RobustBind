import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
from models import PointBind_models
from imagebind.imagebind_model import ModalityType
import logging
from types import SimpleNamespace
import torch_scatter
import abc
from perf.profiling import GpuMemoryTracker
from enum import Enum, auto

MODALITY_TO_MLP = {
    "image":   "mlp_for_image",
    "video":   "mlp_for_video",
    "audio":   "mlp_for_audio",
    "thermal": "mlp_for_thermal",
    "point":   "mlp_for_point",
    "event":   "mlp_for_event",
}

class UniBind(nn.Module):
    def __init__(self, args, use_flash_attention=False, fine_tuned_weights=None, logger=None):
        super(UniBind, self).__init__()
        self.logger = logger or logging.getLogger(__name__)

        self.modality = args.modality
        self.backbone = PointBind_models.PointBind_I2PMAE(use_flash_attention=use_flash_attention)

        state_dict = torch.load(args.pretrain_weights, weights_only=True, map_location='cpu')
        self.backbone.load_state_dict(state_dict, strict=True)
        for param in self.backbone.parameters():
            param.requires_grad_(False)

        # Create modality-specific MLP
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

        if fine_tuned_weights is not None:
            self.logger.info(f"[UniBind init] Loading MLP submodules from '{fine_tuned_weights}'...")
            self.load_fine_tuned_weights(fine_tuned_weights)

    def forward(self, inputs):
        if self.modality == "image":
            outputs = self.__bind(inputs)
            text_embeddings = outputs[ModalityType.TEXT]
            vision_embeddings = self.mlp_for_image(outputs[ModalityType.VISION])
        elif self.modality == "video":
            outputs = self.__bind(inputs)
            text_embeddings = outputs[ModalityType.TEXT]
            vision_embeddings = self.mlp_for_video(outputs[ModalityType.VISION])
        elif self.modality == "audio":
            outputs = self.__bind(inputs)
            text_embeddings = outputs[ModalityType.TEXT]
            vision_embeddings = self.mlp_for_audio(outputs[ModalityType.AUDIO])
        elif self.modality == "thermal":
            outputs = self.__bind(inputs)
            text_embeddings = outputs[ModalityType.TEXT]
            vision_embeddings = self.mlp_for_thermal(outputs[ModalityType.THERMAL])
        elif self.modality == "event":
            outputs = self.__bind(inputs)
            text_embeddings = outputs[ModalityType.TEXT]
            vision_embeddings = self.mlp_for_event(outputs[ModalityType.VISION])
        elif self.modality == "point":
            pc_embeddings = self.backbone.encode_pc(inputs['point'])
            pc_embeddings = self.backbone.bind.modality_head_point(pc_embeddings)
            pc_embeddings = self.backbone.bind.modality_postprocessor_point(pc_embeddings)
            outputs = self.__bind({ModalityType.TEXT: inputs['text']})
            text_embeddings = outputs[ModalityType.TEXT]
            vision_embeddings = self.mlp_for_point(pc_embeddings)

        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        vision_embeddings = vision_embeddings / vision_embeddings.norm(dim=-1, keepdim=True)
        return text_embeddings, vision_embeddings

    def encode_vision(self, inputs):
        if self.modality == "image":
            outputs = self.__bind(inputs)
            vision_embeddings = outputs[ModalityType.VISION]
        elif self.modality == "video":
            outputs = self.__bind(inputs)
            vision_embeddings = outputs[ModalityType.VISION]
        elif self.modality == "audio":
            outputs = self.__bind(inputs)
            vision_embeddings = outputs[ModalityType.AUDIO]
        elif self.modality == "thermal":
            outputs = self.__bind(inputs)
            vision_embeddings = outputs[ModalityType.THERMAL]
        elif self.modality == "event":
            outputs = self.__bind(inputs)
            vision_embeddings = outputs[ModalityType.VISION]
        elif self.modality == "point":
            pc_embeddings = self.backbone.encode_pc(inputs['point'])
            pc_embeddings = self.backbone.bind.modality_head_point(pc_embeddings)
            vision_embeddings = self.backbone.bind.modality_postprocessor_point(pc_embeddings)

        return vision_embeddings / vision_embeddings.norm(dim=-1, keepdim=True)

    def encode_vision_with_mlp(self, inputs):
        if self.modality == "image":
            outputs = self.__bind(inputs)
            vision_embeddings = self.mlp_for_image(outputs[ModalityType.VISION])
        elif self.modality == "video":
            outputs = self.__bind(inputs)
            vision_embeddings = self.mlp_for_video(outputs[ModalityType.VISION])
        elif self.modality == "audio":
            outputs = self.__bind(inputs)
            vision_embeddings = self.mlp_for_audio(outputs[ModalityType.AUDIO])
        elif self.modality == "thermal":
            outputs = self.__bind(inputs)
            vision_embeddings = self.mlp_for_thermal(outputs[ModalityType.THERMAL])
        elif self.modality == "event":
            outputs = self.__bind(inputs)
            vision_embeddings = self.mlp_for_event(outputs[ModalityType.VISION])
        elif self.modality == "point":
            pc_embeddings = self.backbone.encode_pc(inputs['point'])
            pc_embeddings = self.backbone.modality_head_point(pc_embeddings)
            pc_embeddings = self.backbone.modality_postprocessor_point(pc_embeddings)
            vision_embeddings = self.mlp_for_point(pc_embeddings)

        return vision_embeddings / vision_embeddings.norm(dim=-1, keepdim=True)

    def encode_text(self, inputs):
        text_embeddings = self.__bind(inputs)[ModalityType.TEXT]
        return text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

    def save_fine_tuned_weights(self, checkpoint_path: str):
        mlps_dict = {}
        for mod, mlp_attr in MODALITY_TO_MLP.items():
            if hasattr(self, mlp_attr):
                mlp_module = getattr(self, mlp_attr)
                mlps_dict[mlp_attr] = mlp_module.state_dict()
        torch.save(mlps_dict, checkpoint_path)
        self.logger.info(f"[save_modality_mlps] Saved all MLP submodules to '{checkpoint_path}'.")

    def load_fine_tuned_weights(self, checkpoint_path: str, map_location='cpu'):
        mlps_dict = torch.load(checkpoint_path, weights_only=True, map_location=map_location)
        for mlp_attr, state_dict in mlps_dict.items():
            if hasattr(self, mlp_attr):
                getattr(self, mlp_attr).load_state_dict(state_dict)
                self.logger.info(f"[load_fine_tuned_weights] Loaded '{mlp_attr}' from '{checkpoint_path}'.")
            else:
                self.logger.warning(f"[load_fine_tuned_weights] This model has no '{mlp_attr}' attribute. Skipping.")

    def __bind(self, inputs):
        return self.backbone.bind(inputs)

def init_linear_as_identity(linear_layer):
    assert linear_layer.in_features == linear_layer.out_features
    init.eye_(linear_layer.weight)
    nn.init.zeros_(linear_layer.bias)
    return linear_layer

MODALITY_MAP = {
    "image": ModalityType.VISION,
    "video": ModalityType.VISION,
    "audio": ModalityType.AUDIO,
    "thermal": ModalityType.THERMAL,
    "point": ModalityType.POINT,
    "event": ModalityType.VISION
}

class ForwardMode(Enum):
    EMBEDDINGS = auto()
    LOGITS = auto()

class Model(nn.Module):
    @abc.abstractmethod
    def forward(self, x, mode: ForwardMode):
        pass

class UniBindModel(Model):
    def __init__(
        self,
        device,
        pretrain_weights,
        modality,
        centre_embeddings,
        centre_labels,
        label_to_index,
        index_to_label,
        logger=None,
        use_flash_attention=False,
        fine_tuned_weights=None
    ):
        super().__init__()
        self.logger = logger if logger else logging.getLogger(__name__)
        self.logger.info("Initializing UniBindModel...")

        self.unibind = UniBind(
            SimpleNamespace(pretrain_weights=pretrain_weights, modality=modality),
            use_flash_attention=use_flash_attention,
            fine_tuned_weights=fine_tuned_weights,
            logger=self.logger
        )

        self.modality = modality
        self.label_to_index_map = label_to_index
        self.index_to_label_map = index_to_label

        self.logger.info("Storing centre embeddings on device...")
        self.centre_embeddings = centre_embeddings.to(device)

        self.logger.info("Building centre_label_indices...")
        self.centre_label_indices = torch.tensor(
            [self.label_to_index_map[lbl] for lbl in centre_labels],
            dtype=torch.int64,
            device=device
        )
    
    def forward(self, x, mode: ForwardMode):
        if mode == ForwardMode.EMBEDDINGS:
            return self._encode(x)
        elif mode == ForwardMode.LOGITS:
            return self._logits(x)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _logits(self, x):
        embeddings = self._encode(x)
        with GpuMemoryTracker(self.logger):
            similarity = embeddings @ self.centre_embeddings.t()
        with GpuMemoryTracker(self.logger):
            class_scores = torch_scatter.scatter_logsumexp(similarity, self.centre_label_indices, dim=1)
        return class_scores, similarity

    def _encode(self, x):
        modality = MODALITY_MAP[self.modality]
        inp_dict = {modality: x}
        emb = self.unibind.encode_vision_with_mlp(inp_dict)
        return emb / emb.norm(dim=-1, keepdim=True)

    def save_fine_tuned_weights(self, path: str):
        self.logger.info(f"[save_fine_tuned_weights] Saving fine tuned weights to '{path}'...")
        self.unibind.save_fine_tuned_weights(path)

    def load_fine_tuned_weights(self, path: str):
        self.logger.info(f"[load_fine_tuned_weights] Loading fine tuned weights from '{path}'...")
        self.unibind.load_fine_tuned_weights(path)
