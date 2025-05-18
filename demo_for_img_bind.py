import torch
from model import ImageBindModel, ForwardMode
from data_util import load_and_transform_audio_data, load_and_transform_vision_data, load_and_transform_text

# Set your class names and input paths
class_names = ["dog", "car", "bird"]

image_paths = ["assets/image_dog.jpg", "assets/image_chair.jpg"]
audio_paths = ["assets/audio_car.wav", "assets/audio_airplane.wav"]
# thermal_paths = ["assets/dog_ir.jpg", "assets/car_ir.jpg", "assets/bird_ir.jpg"]

device = "cuda" if torch.cuda.is_available() else "cpu"

# Instantiate models for image and audio
image_model = ImageBindModel(device=device, modality="image", class_strings=class_names)
audio_model = ImageBindModel(device=device, modality="audio", class_strings=class_names)
text_model = ImageBindModel(device=device, modality="text", class_strings=class_names)

# Encode each modality and shared text embeddings

image_emb = image_model(load_and_transform_vision_data(image_paths, device), mode=ForwardMode.EMBEDDINGS)
audio_emb = audio_model(load_and_transform_audio_data(audio_paths, device), mode=ForwardMode.EMBEDDINGS)
text_emb = text_model(load_and_transform_text(class_names, device), mode=ForwardMode.EMBEDDINGS)

# Compute normalized similarities
def softmax_sim(a, b):
    return torch.softmax(a @ b.T, dim=-1)

print("Vision x Text:\n", softmax_sim(image_emb, text_emb))
print("Audio x Text:\n", softmax_sim(audio_emb, text_emb))
print("Vision x Audio:\n", softmax_sim(image_emb, audio_emb))
