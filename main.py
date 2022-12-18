"""
Implementing Donut from huggingface
"""

import re
import os
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
from dotenv import load_dotenv

load_dotenv()


def model_prompt(task_, **kwargs):
    if task_ == "CORD":
        model_ = os.getenv(task_)
        task_prompt_ = os.getenv("task_prompt_CORD")
    elif task_ == "RVL-CDIP":
        model_ = os.getenv(task_)
        task_prompt_ = os.getenv("task_prompt_RVL-CDIP")
    elif task_ == "DocVQA":
        question_ = kwargs["question_"]
        model_ = os.getenv(task_)
        task_prompt_ = os.getenv("task_prompt_DocVQA").replace(
            "{user_input}", question_
        )
    else:
        raise ValueError("Not a valid task")
    print(f"model: {model_} task_promt: {task_prompt_}")
    return model_, task_prompt_


def processor_model(model_, task_prompt_, image):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = DonutProcessor.from_pretrained(model_)
    model = VisionEncoderDecoderModel.from_pretrained(model_)
    model.to(device)
    decoder_input_ids = processor.tokenizer(
        task_prompt_, add_special_tokens=False, return_tensors="pt"
    ).input_ids
    decoder_input_ids.to(device)
    pixel_values = processor(image, return_tensors="pt").pixel_values
    pixel_values.to(device)
    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )
    return outputs, processor


def sequence_processor(outputs, processor):
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(
        processor.tokenizer.pad_token, ""
    )
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
    return processor.token2json(sequence)
