import torch
import torch.nn as nn
import torch.nn.init as init
import torch_scatter
import torch.nn.functional as F
import logging
import abc
from enum import Enum, auto
from types import SimpleNamespace

# PointBind / UniBind
from models import PointBind_models
from imagebind.lora import lora_load_state_dict, save_lora_weights, load_lora_weights

# LanguageBind
from languagebind import LanguageBind, LanguageBindImageTokenizer, to_device, transform_dict as lb_transform_dict

# ImageBind
from imagebind.models.imagebind_model import imagebind_model
from imagebind.models import tokenizer as imagebind_tokenizer
from imagebind.imagebind_model import ModalityType as ImageBindModalityType
from imagebind.processing import transform_dict as imagebind_transform_dict

class ForwardMode(Enum):
    EMBEDDINGS = auto()
    LOGITS = auto()

MODALITY_TEMPLATES = {
    "image":   "a photo of a {}",
    "video":   "a video of a {}",
    "depth":   "a depth photo of a {}",
    "thermal": "a photo of a {}",
    "audio":   "a sound of a {}",
    "point":   "a 3D point cloud of a {}",
    "event":   "an event frame of a {}",
}

LANGUAGEBIND_MODEL_NAME_MAP = {
    "video":   "LanguageBind_Video_FT",
    "audio":   "LanguageBind_Audio_FT",
    "thermal": "LanguageBind_Thermal",
    "image":   "LanguageBind_Image",
    "depth":   "LanguageBind_Depth",
}

IMAGEBIND_MODALITY_MAP = {
    "image":   ImageBindModalityType.VISION,
    "video":   ImageBindModalityType.VISION,
    "audio":   ImageBindModalityType.AUDIO,
    "thermal": ImageBindModalityType.THERMAL,
    "depth":   ImageBindModalityType.DEPTH,
    "imu":     ImageBindModalityType.IMU,
}

MODALITY_MAP = {
    "image": ImageBindModalityType.VISION,
    "video": ImageBindModalityType.VISION,
    "audio": ImageBindModalityType.AUDIO,
    "thermal": ImageBindModalityType.THERMAL,
    "point": ImageBindModalityType.POINT,
    "event": ImageBindModalityType.VISION
}

MODALITY_TO_MLP = {
    "image":   "mlp_for_image",
    "video":   "mlp_for_video",
    "audio":   "mlp_for_audio",
    "thermal": "mlp_for_thermal",
    "point":   "mlp_for_point",
    "event":   "mlp_for_event",
}


# ============================ Shared Model Base ============================
class Model(nn.Module):
    @abc.abstractmethod
    def forward(self, x, mode: ForwardMode):
        pass

class UniBind(nn.Module):
    def __init__(self, args, use_flash_attention=False, use_lora=False, lora_rank=4, lora_alpha=8, use_fine_tune=False, lora_weights=None, fine_tuned_weights=None, logger=None):
        super(UniBind, self).__init__()
        self.logger = logger or logging.getLogger(__name__)

        self.modality = args.modality
        self.use_fine_tune = use_fine_tune
        self.backbone = PointBind_models.PointBind_I2PMAE(use_flash_attention=use_flash_attention, use_lora=use_lora, lora_rank=lora_rank, lora_alpha=lora_alpha)

        state_dict = torch.load(args.pretrain_weights, weights_only=True, map_location='cpu')

        if lora_weights is not None:
            self.logger.info(f"[UniBind init] use_lora: {use_lora}")
            self.logger.info(f"[UniBind init] Loading LoRA weights from '{lora_weights}'...")
            lora_state_dict = torch.load(lora_weights, weights_only=True, map_location='cpu')
            state_dict.update(lora_state_dict)
            self.logger.info("[UniBind init] Loaded LoRA weights.")

        lora_load_state_dict(self.backbone, state_dict)

        # Create modality-specific MLP
        if self.modality == "image":
            self.mlp_for_image = init_linear_as_identity(nn.Linear(1024, 1024))
            self.mlp_for_image.requires_grad_(use_fine_tune)
        elif self.modality == "video":
            self.mlp_for_video = init_linear_as_identity(nn.Linear(1024, 1024))
            self.mlp_for_video.requires_grad_(use_fine_tune)
        elif self.modality == "audio":
            self.mlp_for_audio = init_linear_as_identity(nn.Linear(1024, 1024))
            self.mlp_for_audio.requires_grad_(use_fine_tune)
        elif self.modality == "thermal":
            self.mlp_for_thermal = init_linear_as_identity(nn.Linear(1024, 1024))
            self.mlp_for_thermal.requires_grad_(use_fine_tune)
        elif self.modality == "point":
            self.mlp_for_point = init_linear_as_identity(nn.Linear(1024, 1024))
            self.mlp_for_point.requires_grad_(use_fine_tune)
        elif self.modality == "event":
            self.mlp_for_event = init_linear_as_identity(nn.Linear(1024, 1024))
            self.mlp_for_event.requires_grad_(use_fine_tune)
        else:
            raise ValueError(f"Unsupported modality: {self.modality}")

        if fine_tuned_weights is not None:
            self.logger.info(f"[UniBind init] Loading MLP submodules from '{fine_tuned_weights}'...")
            self.load_fine_tuned_weights(fine_tuned_weights)

    def forward(self, inputs):
        if self.modality == "image":
            outputs = self.__bind(inputs)
            text_embeddings = outputs[ImageBindModalityType.TEXT]
            if self.use_fine_tune:
                vision_embeddings = self.mlp_for_image(outputs[ImageBindModalityType.VISION])
            else:
                vision_embeddings = outputs[ImageBindModalityType.VISION]
        elif self.modality == "video":
            outputs = self.__bind(inputs)
            text_embeddings = outputs[ImageBindModalityType.TEXT]
            if self.use_fine_tune:
                vision_embeddings = self.mlp_for_video(outputs[ImageBindModalityType.VISION])
            else:
                vision_embeddings = outputs[ImageBindModalityType.VISION]
        elif self.modality == "audio":
            outputs = self.__bind(inputs)
            text_embeddings = outputs[ImageBindModalityType.TEXT]
            if self.use_fine_tune:
                vision_embeddings = self.mlp_for_audio(outputs[ImageBindModalityType.AUDIO])
            else:
                vision_embeddings = outputs[ImageBindModalityType.AUDIO]
        elif self.modality == "thermal":
            outputs = self.__bind(inputs)
            text_embeddings = outputs[ImageBindModalityType.TEXT]
            if self.use_fine_tune:
                vision_embeddings = self.mlp_for_thermal(outputs[ImageBindModalityType.THERMAL])
            else:
                vision_embeddings = outputs[ImageBindModalityType.THERMAL]
        elif self.modality == "event":
            outputs = self.__bind(inputs)
            text_embeddings = outputs[ImageBindModalityType.TEXT]
            if self.use_fine_tune:
                vision_embeddings = self.mlp_for_event(outputs[ImageBindModalityType.VISION])
            else:
                vision_embeddings = outputs[ImageBindModalityType.VISION]
        elif self.modality == "point":
            pc_embeddings = self.backbone.encode_pc(inputs['point'])
            pc_embeddings = self.backbone.bind.modality_head_point(pc_embeddings)
            pc_embeddings = self.backbone.bind.modality_postprocessor_point(pc_embeddings)
            outputs = self.__bind({ImageBindModalityType.TEXT: inputs['text']})
            text_embeddings = outputs[ImageBindModalityType.TEXT]
            if self.use_fine_tune:
                vision_embeddings = self.mlp_for_point(pc_embeddings)
            else:
                vision_embeddings = pc_embeddings

        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        vision_embeddings = vision_embeddings / vision_embeddings.norm(dim=-1, keepdim=True)
        return text_embeddings, vision_embeddings

    def encode_vision(self, inputs):
        if self.modality == "image":
            outputs = self.__bind(inputs)
            vision_embeddings = outputs[ImageBindModalityType.VISION]
        elif self.modality == "video":
            outputs = self.__bind(inputs)
            vision_embeddings = outputs[ImageBindModalityType.VISION]
        elif self.modality == "audio":
            outputs = self.__bind(inputs)
            vision_embeddings = outputs[ImageBindModalityType.AUDIO]
        elif self.modality == "thermal":
            outputs = self.__bind(inputs)
            vision_embeddings = outputs[ImageBindModalityType.THERMAL]
        elif self.modality == "event":
            outputs = self.__bind(inputs)
            vision_embeddings = outputs[ImageBindModalityType.VISION]
        elif self.modality == "point":
            pc_embeddings = self.backbone.encode_pc(inputs['point'])
            pc_embeddings = self.backbone.bind.modality_head_point(pc_embeddings)
            vision_embeddings = self.backbone.bind.modality_postprocessor_point(pc_embeddings)

        return vision_embeddings / vision_embeddings.norm(dim=-1, keepdim=True)

    def encode_vision_with_mlp(self, inputs):
        if self.modality == "image":
            outputs = self.__bind(inputs)
            if self.use_fine_tune:
                vision_embeddings = self.mlp_for_image(outputs[ImageBindModalityType.VISION])
            else:
                vision_embeddings = outputs[ImageBindModalityType.VISION]
        elif self.modality == "video":
            outputs = self.__bind(inputs)
            if self.use_fine_tune:
                vision_embeddings = self.mlp_for_video(outputs[ImageBindModalityType.VISION])
            else:
                vision_embeddings = outputs[ImageBindModalityType.VISION]
        elif self.modality == "audio":
            outputs = self.__bind(inputs)
            if self.use_fine_tune:
                vision_embeddings = self.mlp_for_audio(outputs[ImageBindModalityType.AUDIO])
            else:
                vision_embeddings = outputs[ImageBindModalityType.AUDIO]
        elif self.modality == "thermal":
            outputs = self.__bind(inputs)
            if self.use_fine_tune:
                vision_embeddings = self.mlp_for_thermal(outputs[ImageBindModalityType.THERMAL])
            else:
                vision_embeddings = outputs[ImageBindModalityType.THERMAL]
        elif self.modality == "event":
            outputs = self.__bind(inputs)
            if self.use_fine_tune:
                vision_embeddings = self.mlp_for_event(outputs[ImageBindModalityType.VISION])
            else:
                vision_embeddings = outputs[ImageBindModalityType.VISION]
        elif self.modality == "point":
            pc_embeddings = self.backbone.encode_pc(inputs['point'])
            pc_embeddings = self.backbone.bind.modality_head_point(pc_embeddings)
            pc_embeddings = self.backbone.bind.modality_postprocessor_point(pc_embeddings)
            if self.use_fine_tune:
                vision_embeddings = self.mlp_for_point(pc_embeddings)
            else:
                vision_embeddings = pc_embeddings

        return vision_embeddings / vision_embeddings.norm(dim=-1, keepdim=True)

    def encode_text(self, inputs):
        text_embeddings = self.__bind(inputs)[ImageBindModalityType.TEXT]
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

    def load_lora_weights(self, checkpoint_path: str):
        self.logger.info(f"[load_lora_weights] Loading LoRA weights from '{checkpoint_path}'...")
        load_lora_weights(self.backbone, checkpoint_path)
        self.logger.info("[load_lora_weights] Loaded LoRA weights.")

    def save_lora_weights(self, checkpoint_path: str):
        self.logger.info(f"[save_lora_weights] Saving LoRA weights to '{checkpoint_path}'...")
        save_lora_weights(self.backbone, checkpoint_path)

    def __bind(self, inputs):
        return self.backbone.bind(inputs)

def init_linear_as_identity(linear_layer):
    assert linear_layer.in_features == linear_layer.out_features
    init.eye_(linear_layer.weight)
    nn.init.zeros_(linear_layer.bias)
    return linear_layer

# ============================ UniBindModel ============================
# (as defined in your full version, unchanged)
# Please assume your full UniBindModel code is already here.

class UniBindModel(Model):
    def __init__(
        self,
        device,
        pretrain_weights,
        modality,
        centre_embeddings,
        centre_labels,
        label_to_index,
        logger=None,
        use_flash_attention=False,
        use_lora=False,
        lora_rank=4,
        lora_alpha=8,
        use_fine_tune=False,
        lora_weights=None,
        fine_tuned_weights=None,
    ):
        super().__init__()
        self.logger = logger if logger else logging.getLogger(__name__)
        self.logger.info("Initializing UniBindModel...")
        self.logger.info(f"Use LoRa: {use_lora}, LoRa rank: {lora_rank}, LoRa alpha: {lora_alpha}")

        self.unibind = UniBind(
            SimpleNamespace(pretrain_weights=pretrain_weights, modality=modality),
            use_flash_attention=use_flash_attention,
            fine_tuned_weights=fine_tuned_weights,
            lora_weights=lora_weights,
            logger=self.logger,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            use_fine_tune=use_fine_tune,
        )

        self.modality = modality
        self.label_to_index_map = label_to_index

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

    def _logits(self, x, temperature=1000.0):
        embeddings = self._encode(x)
        similarity = embeddings @ self.centre_embeddings.t()
        class_raw_scores = torch_scatter.scatter_logsumexp(similarity * temperature, self.centre_label_indices, dim=1)
        return class_raw_scores / temperature, similarity

    def _encode(self, x):
        modality = MODALITY_MAP[self.modality]
        inp_dict = {modality: x}
        emb = self.unibind.encode_vision_with_mlp(inp_dict)
        return emb / emb.norm(dim=-1, keepdim=True)
    
    def save_lora_weights(self, path: str):
        self.logger.info(f"[save_lora_weights] Saving LoRA weights to '{path}'...")
        self.unibind.save_lora_weights(path)

    def load_lora_weights(self, path: str):
        self.logger.info(f"[load_lora_weights] Loading LoRA weights from '{path}'...")
        self.unibind.load_lora_weights(path)

    def save_fine_tuned_weights(self, path: str):
        self.logger.info(f"[save_fine_tuned_weights] Saving fine tuned weights to '{path}'...")
        self.unibind.save_fine_tuned_weights(path)

    def load_fine_tuned_weights(self, path: str):
        self.logger.info(f"[load_fine_tuned_weights] Loading fine tuned weights from '{path}'...")
        self.unibind.load_fine_tuned_weights(path)


# ============================ LanguageBindModel ============================
class LanguageBindModel(Model):
    def __init__(self, device, modality, class_strings, logger=None):
        super().__init__()
        self.device = device
        self.modality = modality
        self.class_strings = class_strings
        self.logger = logger or logging.getLogger(__name__)

        template = MODALITY_TEMPLATES.get(modality, "a photo of a {}")
        prompts = [template.format(cls) for cls in class_strings]

        model_name = LANGUAGEBIND_MODEL_NAME_MAP[modality]
        self.languagebind = LanguageBind(clip_type={modality: model_name}).to(device).eval()
        tokenizer = LanguageBindImageTokenizer.from_pretrained(model_name)

        tokens = tokenizer(prompts, max_length=77, padding="max_length", truncation=True, return_tensors="pt")
        tokens = to_device(tokens, device)

        with torch.no_grad():
            text_embs = self.languagebind.encode_text(tokens)
        self.class_embeddings = text_embs / text_embs.norm(dim=-1, keepdim=True)
        self.transform = lb_transform_dict[modality](self.languagebind.modality_config[modality])

    def forward(self, x, mode: ForwardMode):
        if mode == ForwardMode.EMBEDDINGS:
            return self._encode(x)
        elif mode == ForwardMode.LOGITS:
            return self._logits(x)
        raise ValueError(f"Unknown mode: {mode}")

    def _encode(self, x):
        if isinstance(x, list):
            x = self.transform(x)
        x = to_device(x, self.device)
        with torch.no_grad():
            emb = self.languagebind({self.modality: x})[self.modality]
        return emb / emb.norm(dim=-1, keepdim=True)

    def _logits(self, x, temperature=100.0):
        emb = self._encode(x)
        logits = emb @ self.class_embeddings.T
        return logits / temperature, logits


# ============================ ImageBindModel ============================
class ImageBindModel(Model):
    def __init__(self, device, modality, class_strings, logger=None):
        super().__init__()
        self.device = device
        self.modality = modality
        self.class_strings = class_strings
        self.logger = logger or logging.getLogger(__name__)

        template = MODALITY_TEMPLATES.get(modality, "a photo of a {}")
        prompts = [template.format(cls) for cls in class_strings]

        self.model = imagebind_model(pretrained=True).to(device).eval()
        tokens = imagebind_tokenizer.tokenize(prompts).to(device)

        with torch.no_grad():
            text_embs = self.model({ImageBindModalityType.TEXT: tokens})[ImageBindModalityType.TEXT]
        self.class_embeddings = text_embs / text_embs.norm(dim=-1, keepdim=True)
        self.transform = imagebind_transform_dict[modality]

    def forward(self, x, mode: ForwardMode):
        if mode == ForwardMode.EMBEDDINGS:
            return self._encode(x)
        elif mode == ForwardMode.LOGITS:
            return self._logits(x)
        raise ValueError(f"Unknown mode: {mode}")

    def _encode(self, x):
        if isinstance(x, list):
            x = self.transform(x)
        x = x.to(self.device)
        with torch.no_grad():
            emb = self.model({IMAGEBIND_MODALITY_MAP[self.modality]: x})[IMAGEBIND_MODALITY_MAP[self.modality]]
        return emb / emb.norm(dim=-1, keepdim=True)

    def _logits(self, x, temperature=100.0):
        emb = self._encode(x)
        logits = emb @ self.class_embeddings.T
        return logits / temperature, logits
