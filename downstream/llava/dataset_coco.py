import os, json, random, torch
from torch.utils.data import Dataset
from PIL import Image
from downstream.llava.conversation import conv_templates
from downstream.llava.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX
from downstream.llava.mm_utils import tokenizer_image_token

class COCOCaptionDataset(Dataset):
    def __init__(self, json_path, image_root, tokenizer, image_processor, max_samples=None, debug=False):
        with open(json_path, "r") as f:
            self.data = json.load(f)

        if max_samples is not None and max_samples < len(self.data):
            indices = torch.arange(max_samples) if debug else torch.randperm(len(self.data))[:max_samples]
            self.data = [self.data[i] for i in indices]

        self.image_root = image_root
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.image_root, item["image"])
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

        # === Build LLaVA-style conversation ===
        caption = random.choice(item["captions"])
        conv = conv_templates["llava_v1"].copy()
        user_message = f"{DEFAULT_IMAGE_TOKEN}\nDescribe the image."
        conv.append_message(conv.roles[0], user_message)
        conv.append_message(conv.roles[1], caption)
        prompt = conv.get_prompt()

        # === Tokenize full prompt ===
        input_ids = tokenizer_image_token(prompt, self.tokenizer, return_tensors="pt")
        labels = input_ids.clone()

        # === Mask all tokens before the assistant reply ===
        # Tokenize only up to assistant message to get instruction length
        sep = conv.sep  # usually '</s>'
        assistant_tag = conv.roles[1] + ": "  # 'ASSISTANT: '

        instruction_only = conv.roles[0] + ": " + user_message + sep + assistant_tag
        instruction_len = len(tokenizer_image_token(instruction_only, self.tokenizer))

        # Mask everything before the caption
        labels[:instruction_len] = IGNORE_INDEX
        labels[input_ids == self.tokenizer.pad_token_id] = IGNORE_INDEX

        return {
            "input_ids": input_ids,
            "labels": labels,
            "images": image_tensor,
        }

    def collate_fn(self, batch):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [x["input_ids"] for x in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            [x["labels"] for x in batch], batch_first=True, padding_value=IGNORE_INDEX)
        images = torch.stack([x["images"] for x in batch])
        return {"input_ids": input_ids, "labels": labels, "images": images}
