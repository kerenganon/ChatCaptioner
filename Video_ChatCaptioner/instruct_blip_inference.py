from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import requests
import gc

# PROMPT = """
#     If there are people in the frame, describe what they are doing in a short sentence.
#     If there aren't people, say that there aren't any people.
# """
PROMPT = """
    Write a caption of the image focusing on what the people in it are doing in a short sentence.
    If there are no people, caption the frame.
"""

PROMPT_ONLY_PEOPLE = """
    Write a caption of the image focusing on what the people in it are doing in a short sentence.
    If there are no people, say that there aren't any people.
"""

# taken from https://github.com/NielsRogge/Transformers-Tutorials/blob/master/InstructBLIP/Inference_with_InstructBLIP.ipynb
class InstructBlipInferencer:
    def __init__(self):
        gc.collect()
        torch.cuda.empty_cache()
        self.model = InstructBlipForConditionalGeneration.from_pretrained(
            "Salesforce/instructblip-flan-t5-xl"
        )
        self.processor = InstructBlipProcessor.from_pretrained(
            "Salesforce/instructblip-flan-t5-xl"
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def call_instruct_blip(self, image, text=PROMPT_ONLY_PEOPLE):
        # image = Image.open(frame_path).convert("RGB")
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(
            self.device
        )

        outputs = self.model.generate(
            **inputs,
            do_sample=False,
            num_beams=5,
            max_length=35,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=0,
            temperature=1,
        )
        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[
            0
        ].strip()
        return generated_text
