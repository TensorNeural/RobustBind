import torch
from model import ForwardMode, LanguageBindClassifier, LANGUAGEBIND_MODEL_NAME_MAP, LANGUAGEBIND_TOKENIZER_MAP, LANGUAGEBIND_TOKENIZER_NAME_MAP
from binds.languagebind import to_device,  transform_dict as lb_transform_dict, LanguageBindImageTokenizer

# Class labels and input paths
class_names = ["dog", "car", "bird"]
image_paths = ["assets/image_dog.jpg", "assets/image_chair.jpg"]
audio_paths = ["assets/audio_car.wav", "assets/audio_airplane.wav"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate models for image and audio
image_model = LanguageBindClassifier(device=device, modality="image", class_strings=class_names)
audio_model = LanguageBindClassifier(device=device, modality="audio", class_strings=class_names)

# === Separate text encoder logic ===
# Use one tokenizer for all prompts (assuming text prompts shared across modalities)
tokenizer_modality = "image"  # or "audio", depending on your use case
model_name = LANGUAGEBIND_MODEL_NAME_MAP[tokenizer_modality]
tokenizer_class = LANGUAGEBIND_TOKENIZER_MAP[tokenizer_modality]
tokenizer = tokenizer_class.from_pretrained(LANGUAGEBIND_TOKENIZER_NAME_MAP[tokenizer_modality], cache_dir=".cache/tokenizer")
tokenized_text = tokenizer(class_names, max_length=77, padding="max_length", truncation=True, return_tensors="pt")
text_tokens = to_device(tokenized_text, device)

# Use one LanguageBind instance for encode_text (any modality will do, image is safe)
with torch.no_grad():
    text_emb = image_model.encode_text(text_tokens)

# === Encode image and audio ===
image_transform = lb_transform_dict["image"](image_model.modality_config())
audio_transform = lb_transform_dict["audio"](audio_model.modality_config())
image_tensor = to_device(image_transform(image_paths), device)
audio_tensor = to_device(audio_transform(audio_paths), device)

image_emb = image_model(image_tensor, mode=ForwardMode.EMBEDDINGS)
audio_emb = audio_model(audio_tensor, mode=ForwardMode.EMBEDDINGS)

# === Similarity function ===
def softmax_sim(a, b):
    return torch.softmax(a @ b.T, dim=-1)

# === Print similarities ===
print("Vision x Text:\n", softmax_sim(image_emb, text_emb).detach().cpu().numpy())
print("Audio x Text:\n", softmax_sim(audio_emb, text_emb).detach().cpu().numpy())
print("Vision x Audio:\n", softmax_sim(image_emb, audio_emb).detach().cpu().numpy())
