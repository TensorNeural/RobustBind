import torch
import torch.nn as nn
import torch.nn.init as init
import torch_scatter
import torch.nn.functional as F
import logging
import abc
from enum import Enum, auto
from types import SimpleNamespace
# import traceback

# PointBind / UniBind
from models import PointBind_models
from imagebind.lora import lora_load_state_dict, save_lora_weights, load_lora_weights

from binds.languagebind import (
    LanguageBind, to_device,
    LanguageBindImageTokenizer, LanguageBindVideoTokenizer, LanguageBindAudioTokenizer, 
    LanguageBindThermalTokenizer, LanguageBindDepthTokenizer
)

# ImageBind
from binds.imagebind.models import imagebind_model
from data_util import load_and_transform_text
from binds.imagebind.models.imagebind_model import ModalityType as ImageBindModalityType
from imagebind.imagebind_model import ModalityType
from shared_types import Modality

class ForwardMode(Enum):
    EMBEDDINGS = auto()
    LOGITS = auto()

MODALITY_TEMPLATES = {
    Modality.IMAGE:   "a photo of a {}",
    Modality.VIDEO:   "a video of a {}",
    Modality.DEPTH:   "a depth photo of a {}",
    Modality.THERMAL: "a photo of a {}",
    Modality.AUDIO:   "a sound of a {}",
    Modality.POINT:   "a 3D point cloud of a {}",
    Modality.EVENT:   "an event frame of a {}",
}

LANGUAGEBIND_MODEL_NAME_MAP = {
    Modality.VIDEO:   "LanguageBind_Video_FT",
    Modality.AUDIO:   "LanguageBind_Audio_FT",
    Modality.THERMAL: "LanguageBind_Thermal",
    Modality.IMAGE:   "LanguageBind_Image",
    Modality.DEPTH:   "LanguageBind_Depth",
}
LANGUAGEBIND_TOKENIZER_MAP = {
    Modality.IMAGE:   LanguageBindImageTokenizer,
    Modality.VIDEO:   LanguageBindVideoTokenizer,
    Modality.AUDIO:   LanguageBindAudioTokenizer,
    Modality.THERMAL: LanguageBindThermalTokenizer,
    Modality.DEPTH:   LanguageBindDepthTokenizer,
}
LANGUAGEBIND_TOKENIZER_NAME_MAP = {
    Modality.IMAGE:   "lb203/LanguageBind_Image",
    Modality.VIDEO:   "lb203/LanguageBind_Video",
    Modality.AUDIO:   "lb203/LanguageBind_Audio",
    Modality.THERMAL: "lb203/LanguageBind_Thermal",
    Modality.DEPTH:   "lb203/LanguageBind_Depth",
}
IMAGEBIND_MODALITY_MAP = {
    Modality.TEXT:    ImageBindModalityType.TEXT,
    Modality.IMAGE:   ImageBindModalityType.VISION,
    Modality.VIDEO:   ImageBindModalityType.VISION,
    Modality.AUDIO:   ImageBindModalityType.AUDIO,
    Modality.THERMAL: ImageBindModalityType.THERMAL,
    Modality.DEPTH:   ImageBindModalityType.DEPTH,
}

MODALITY_TO_MLP = {
    Modality.IMAGE:   "mlp_for_image",
    Modality.VIDEO:   "mlp_for_video",
    Modality.AUDIO:   "mlp_for_audio",
    Modality.THERMAL: "mlp_for_thermal",
    Modality.POINT:   "mlp_for_point",
    Modality.EVENT:   "mlp_for_event",
}

MODALITY_MAP = {
    Modality.IMAGE: ModalityType.VISION,
    Modality.VIDEO: ModalityType.VISION,
    Modality.AUDIO: ModalityType.AUDIO,
    Modality.THERMAL: ModalityType.THERMAL,
    Modality.EVENT: ModalityType.VISION,
    Modality.POINT: ModalityType.POINT,
    Modality.TEXT: ModalityType.TEXT,
}


# ============================ Shared Model Base ============================
class Model(nn.Module):
    @abc.abstractmethod
    def forward(self, x, mode: ForwardMode):
        pass

    def extract_tensor(self, x):
        pass

    def wrap_tensor(self, x):
        pass
    
    def data_to_device(self, x, device):
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
        if self.modality == Modality.IMAGE:
            self.mlp_for_image = init_linear_as_identity(nn.Linear(1024, 1024))
            self.mlp_for_image.requires_grad_(use_fine_tune)
        elif self.modality == Modality.VIDEO:
            self.mlp_for_video = init_linear_as_identity(nn.Linear(1024, 1024))
            self.mlp_for_video.requires_grad_(use_fine_tune)
        elif self.modality == Modality.AUDIO:
            self.mlp_for_audio = init_linear_as_identity(nn.Linear(1024, 1024))
            self.mlp_for_audio.requires_grad_(use_fine_tune)
        elif self.modality == Modality.THERMAL:
            self.mlp_for_thermal = init_linear_as_identity(nn.Linear(1024, 1024))
            self.mlp_for_thermal.requires_grad_(use_fine_tune)
        elif self.modality == Modality.POINT:
            self.mlp_for_point = init_linear_as_identity(nn.Linear(1024, 1024))
            self.mlp_for_point.requires_grad_(use_fine_tune)
        elif self.modality == Modality.EVENT:
            self.mlp_for_event = init_linear_as_identity(nn.Linear(1024, 1024))
            self.mlp_for_event.requires_grad_(use_fine_tune)
        else:
            raise ValueError(f"Unsupported modality: {self.modality}")

        if fine_tuned_weights is not None:
            self.logger.info(f"[UniBind init] Loading MLP submodules from '{fine_tuned_weights}'...")
            self.load_fine_tuned_weights(fine_tuned_weights)

    def forward(self, inputs):
        if self.modality == Modality.IMAGE:
            outputs = self.__bind(inputs)
            text_embeddings = outputs[ImageBindModalityType.TEXT]
            if self.use_fine_tune:
                vision_embeddings = self.mlp_for_image(outputs[ImageBindModalityType.VISION])
            else:
                vision_embeddings = outputs[ImageBindModalityType.VISION]
        elif self.modality == Modality.VIDEO:
            outputs = self.__bind(inputs)
            text_embeddings = outputs[ImageBindModalityType.TEXT]
            if self.use_fine_tune:
                vision_embeddings = self.mlp_for_video(outputs[ImageBindModalityType.VISION])
            else:
                vision_embeddings = outputs[ImageBindModalityType.VISION]
        elif self.modality == Modality.AUDIO:
            outputs = self.__bind(inputs)
            text_embeddings = outputs[ImageBindModalityType.TEXT]
            if self.use_fine_tune:
                vision_embeddings = self.mlp_for_audio(outputs[ImageBindModalityType.AUDIO])
            else:
                vision_embeddings = outputs[ImageBindModalityType.AUDIO]
        elif self.modality == Modality.THERMAL:
            outputs = self.__bind(inputs)
            text_embeddings = outputs[ImageBindModalityType.TEXT]
            if self.use_fine_tune:
                vision_embeddings = self.mlp_for_thermal(outputs[ImageBindModalityType.THERMAL])
            else:
                vision_embeddings = outputs[ImageBindModalityType.THERMAL]
        elif self.modality == Modality.EVENT:
            outputs = self.__bind(inputs)
            text_embeddings = outputs[ImageBindModalityType.TEXT]
            if self.use_fine_tune:
                vision_embeddings = self.mlp_for_event(outputs[ImageBindModalityType.VISION])
            else:
                vision_embeddings = outputs[ImageBindModalityType.VISION]
        elif self.modality == Modality.POINT:
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
        if self.modality == Modality.IMAGE:
            outputs = self.__bind(inputs)
            vision_embeddings = outputs[ImageBindModalityType.VISION]
        elif self.modality == Modality.VIDEO:
            outputs = self.__bind(inputs)
            vision_embeddings = outputs[ImageBindModalityType.VISION]
        elif self.modality == Modality.AUDIO:
            outputs = self.__bind(inputs)
            vision_embeddings = outputs[ImageBindModalityType.AUDIO]
        elif self.modality == Modality.THERMAL:
            outputs = self.__bind(inputs)
            vision_embeddings = outputs[ImageBindModalityType.THERMAL]
        elif self.modality == Modality.EVENT:
            outputs = self.__bind(inputs)
            vision_embeddings = outputs[ImageBindModalityType.VISION]
        elif self.modality == Modality.POINT:
            pc_embeddings = self.backbone.encode_pc(inputs['point'])
            pc_embeddings = self.backbone.bind.modality_head_point(pc_embeddings)
            vision_embeddings = self.backbone.bind.modality_postprocessor_point(pc_embeddings)

        return vision_embeddings / vision_embeddings.norm(dim=-1, keepdim=True)

    def encode_vision_with_mlp(self, inputs):
        if self.modality == Modality.IMAGE:
            outputs = self.__bind(inputs)
            if self.use_fine_tune:
                vision_embeddings = self.mlp_for_image(outputs[ImageBindModalityType.VISION])
            else:
                vision_embeddings = outputs[ImageBindModalityType.VISION]
        elif self.modality == Modality.VIDEO:
            outputs = self.__bind(inputs)
            if self.use_fine_tune:
                vision_embeddings = self.mlp_for_video(outputs[ImageBindModalityType.VISION])
            else:
                vision_embeddings = outputs[ImageBindModalityType.VISION]
        elif self.modality == Modality.AUDIO:
            outputs = self.__bind(inputs)
            if self.use_fine_tune:
                vision_embeddings = self.mlp_for_audio(outputs[ImageBindModalityType.AUDIO])
            else:
                vision_embeddings = outputs[ImageBindModalityType.AUDIO]
        elif self.modality == Modality.THERMAL:
            outputs = self.__bind(inputs)
            if self.use_fine_tune:
                vision_embeddings = self.mlp_for_thermal(outputs[ImageBindModalityType.THERMAL])
            else:
                vision_embeddings = outputs[ImageBindModalityType.THERMAL]
        elif self.modality == Modality.EVENT:
            outputs = self.__bind(inputs)
            if self.use_fine_tune:
                vision_embeddings = self.mlp_for_event(outputs[ImageBindModalityType.VISION])
            else:
                vision_embeddings = outputs[ImageBindModalityType.VISION]
        elif self.modality == Modality.POINT:
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

# ============================ UniBindClassifier ============================
# (as defined in your full version, unchanged)
# Please assume your full UniBindClassifier code is already here.

class UniBindClassifier(Model):
    def __init__(
        self,
        device,
        pretrain_weights,
        modality,
        centre_embeddings=None,
        centre_labels = None,
        label_to_index = None,
        logger=None,
        use_flash_attention=False,
        use_lora=False,
        lora_rank=4,
        lora_alpha=8,
        use_fine_tune=False,
        lora_weights=None,
        fine_tuned_weights=None,
        use_masked_logsumexp=False,
    ):
        super().__init__()
        self.logger = logger if logger else logging.getLogger(__name__)
        self.logger.info("Initializing UniBindClassifier...")
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
        self.use_masked_logsumexp = use_masked_logsumexp

        if centre_embeddings is not None:
            self.logger.info("Storing centre embeddings on device...")
            self.centre_embeddings = centre_embeddings.to(device)

        if centre_labels is not None:
            self.logger.info("Building centre_label_indices...")
            self.centre_label_indices = torch.tensor(
                [self.label_to_index_map[lbl] for lbl in centre_labels],
                dtype=torch.int64,
                device=device
            )

            # Precompute and store the (C, N) binary mask
            self.num_classes = len(self.label_to_index_map)
            mask = F.one_hot(self.centre_label_indices, num_classes=self.num_classes).T.bool()
            self.register_buffer("centre_class_mask", mask)
    
    def forward(self, x, mode: ForwardMode):
        if mode == ForwardMode.EMBEDDINGS:
            return self._encode(x)
        elif mode == ForwardMode.LOGITS:
            return self._logits(x)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def extract_tensor(self, x):
        return x
    
    def wrap_tensor(self, x):
        return x
    
    def data_to_device(self, x, device):
        return x.to(device)
    
    def encode_text(self, x):
        input_dict = {ModalityType.TEXT: x}
        return self.unibind.encode_text(input_dict)

    def _logits(self, x, temperature=1000.0):
        embeddings = self._encode(x)
        similarity = embeddings @ self.centre_embeddings.t()
        logits = self._compute_class_logits(similarity, temperature)
        return logits, similarity

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
    
    def _compute_class_logits(self, similarity: torch.Tensor, temperature: float) -> torch.Tensor:
        if self.use_masked_logsumexp:
            return self._masked_logsumexp(similarity, temperature)
        else:
            return self._scatter_logsumexp(similarity, temperature)
    
    def _masked_logsumexp(self, similarity: torch.Tensor, temperature: float) -> torch.Tensor:
        B, N = similarity.shape
        C = self.num_classes
        mask = self.centre_class_mask  # (C, N) precomputed

        similarity_exp = similarity.unsqueeze(1)             # (B, 1, N)
        mask_exp = mask.unsqueeze(0).expand(B, C, N)         # (B, C, N)
        masked = similarity_exp.masked_fill(~mask_exp, -1e9) # (B, C, N)
        return torch.logsumexp(masked * temperature, dim=2) / temperature  # (B, C)
    
    def _scatter_logsumexp(self, similarity: torch.Tensor, temperature: float) -> torch.Tensor:
        class_raw_scores = torch_scatter.scatter_logsumexp(similarity * temperature, self.centre_label_indices, dim=1)
        return class_raw_scores / temperature

# ============================ LanguageBindClassifier ============================
class LanguageBindClassifier(Model):
    def __init__(self, device, modality, class_strings, logger=None):
        super().__init__()
        self.device = device
        self.modality = modality
        self.class_strings = class_strings
        self.logger = logger or logging.getLogger(__name__)

        # Prepare prompts using modality-specific templates
        template = MODALITY_TEMPLATES.get(modality, "a {}")
        prompts = [template.format(cls) for cls in class_strings]

        # Load single-modality LanguageBind model and tokenizer
        model_name = LANGUAGEBIND_MODEL_NAME_MAP[modality]
        self.languagebind = LanguageBind(clip_type={modality.value: model_name}, cache_dir='.cache')
        self.languagebind = self.languagebind.to(device)
        self.languagebind.eval()

        tokenizer_class = LANGUAGEBIND_TOKENIZER_MAP[modality]
        tokenizer = tokenizer_class.from_pretrained(
            LANGUAGEBIND_TOKENIZER_NAME_MAP[self.modality],
            cache_dir="./cache/tokenizer"
        )

        # Tokenize class prompts → encode → normalize
        tokens = tokenizer(prompts, max_length=77, padding="max_length", truncation=True, return_tensors="pt")
        tokens = to_device(tokens, device)
        text_embs = self.encode_text(tokens)

        self.class_embeddings = text_embs

    def forward(self, x, mode: ForwardMode):
        if mode == ForwardMode.EMBEDDINGS:
            return self._encode(x)
        elif mode == ForwardMode.LOGITS:
            return self._logits(x)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def extract_tensor(self, x):
        if self.modality == Modality.IMAGE:
            return x["pixel_values"]
        elif self.modality == Modality.VIDEO:
            return x["pixel_values"]
        elif self.modality == Modality.AUDIO:
            return x["pixel_values"]
        elif self.modality == Modality.THERMAL:
            return x["pixel_values"]
        elif self.modality == Modality.DEPTH:
            return x["pixel_values"]
        elif self.modality == Modality.EVENT:
            return x["pixel_values"]
        elif self.modality == Modality.POINT:
            return x["point"]
        else:
            raise ValueError(f"Unknown modality: {self.modality}")
    
    def wrap_tensor(self, x_tensor):
        if self.modality == Modality.IMAGE:
            return {"pixel_values": x_tensor}
        elif self.modality == Modality.VIDEO:
            return {"pixel_values": x_tensor}
        elif self.modality == Modality.AUDIO:
            return {"pixel_values": x_tensor}
        elif self.modality == Modality.THERMAL:
            return {"pixel_values": x_tensor}
        elif self.modality == Modality.DEPTH:
            return {"pixel_values": x_tensor}
        elif self.modality == Modality.EVENT:
            return {"pixel_values": x_tensor}
        elif self.modality == Modality.POINT:
            return {"point": x_tensor}
        else:
            raise ValueError(f"Unknown modality: {self.modality}")
    
    def data_to_device(self, x, device):
        return to_device(x, device)

    def encode_text(self, x):
        # x is already transformed and on device
        with torch.no_grad():
            emb = self.languagebind({'language': x})['language']

        return emb / emb.norm(dim=-1, keepdim=True)

    def modality_config(self):
        return self.languagebind.modality_config[self.modality.value]

    def _encode(self, x):
        # x is already transformed and on device
        emb = self.languagebind({self.modality.value: x})[self.modality.value]
        return emb / emb.norm(dim=-1, keepdim=True)

    def _logits(self, x, temperature=100.0):
        emb = self._encode(x)
        logits = emb @ self.class_embeddings.T
        return logits / temperature, logits

# ============================ ImageBindClassifier ============================
class ImageBindClassifier(Model):
    def __init__(self, device, modality, class_strings, logger=None):
        super().__init__()
        self.device = device
        self.modality = modality
        self.class_strings = class_strings
        self.logger = logger or logging.getLogger(__name__)

        template = MODALITY_TEMPLATES.get(modality, "a {}")
        text_list = [template.format(cls) for cls in class_strings]

        self.model = imagebind_model.imagebind_huge(pretrained=True).to(device)
        self.model.eval()
        tokens = load_and_transform_text(text_list, device)

        with torch.no_grad():
            text_embs = self.model({ImageBindModalityType.TEXT: tokens})[ImageBindModalityType.TEXT]

        self.class_embeddings = text_embs / text_embs.norm(dim=-1, keepdim=True)

    def forward(self, x, mode: ForwardMode):
        if mode == ForwardMode.EMBEDDINGS:
            return self._encode(x)
        elif mode == ForwardMode.LOGITS:
            return self._logits(x)
        raise ValueError(f"Unknown mode: {mode}")

    def extract_tensor(self, x):
        return x
    
    def wrap_tensor(self, x_tensor):
        return x_tensor

    def data_to_device(self, x, device):
        return x.to(device)

    def _encode(self, x):
        imagebind_modality = IMAGEBIND_MODALITY_MAP[self.modality]
        emb = self.model({IMAGEBIND_MODALITY_MAP[self.modality]: x})[imagebind_modality]
        return emb / emb.norm(dim=-1, keepdim=True)

    def _logits(self, x, temperature=100.0):
        emb = self._encode(x)
        logits = emb @ self.class_embeddings.T
        return logits / temperature, logits
