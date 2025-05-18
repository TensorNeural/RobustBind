from enum import Enum

class BindModelType(Enum):
    UNIBIND = "UniBind"
    LANGUAGEBIND = "LanguageBind"
    IMAGEBIND = "ImageBind"

class Modality(Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    THERMAL = "thermal"
    DEPTH = "depth"
    POINT = "point"
    EVENT = "event"
    TEXT = "text"