import torch
from langchain_core.tools import tool
from PIL import Image
from transformers import ViltForQuestionAnswering, ViltProcessor

checkpoint = "dandelin/vilt-b32-finetuned-vqa"
processor = ViltProcessor.from_pretrained(checkpoint)
model = ViltForQuestionAnswering.from_pretrained(checkpoint).eval()


@tool("vision-qa")
def vision_qa(image_path: str, question: str) -> str:
    """Answer a question about an image file."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, question, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    return model.config.id2label[logits.argmax(-1).item()]
