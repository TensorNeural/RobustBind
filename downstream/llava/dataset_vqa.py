import os, json, torch
from torch.utils.data import Dataset
from PIL import Image
from downstream.llava.conversation import conv_templates
from downstream.llava.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX
from downstream.llava.mm_utils import tokenizer_image_token

class VQADataset(Dataset):
    def __init__(self, json_path, image_root, tokenizer, image_processor):
        self.data = json.load(open(json_path))
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.image_root, item["image"])
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

        question = item["question"]
        answer = max(set(item["answers"]), key=item["answers"].count)

        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\nQuestion: {question}\nAnswer:")
        conv.append_message(conv.roles[1], answer)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, return_tensors="pt")
        labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels, "images": image_tensor}

    def collate_fn(self, batch):
        input_ids = torch.nn.utils.rnn.pad_sequence([x["input_ids"] for x in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence([x["labels"] for x in batch], batch_first=True, padding_value=IGNORE_INDEX)
        images = torch.stack([x["images"] for x in batch])
        return {"input_ids": input_ids, "labels": labels, "images": images}
