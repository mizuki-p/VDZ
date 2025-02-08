import os
os.environ['CUDA_VISIBLE_DEVICES'] = '9'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import argparse
import torch
import json
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images

from dataloader import RadDataset, SlakeDataset, OvqaDataset, get_dataset_constructor
from transformers import set_seed
from peft import PeftModel
from tokenizers import AddedToken

torch.set_printoptions(precision=3, threshold=10000, linewidth=225)

def eval_model(args):
    set_seed(0)
    # Model
    disable_torch_init()
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path='downloaded/llava-med-v1.5-mistral-7b',
        model_base=None,
        model_name='llava-med-v1.5-mistral-7b'
    )
    model = PeftModel.from_pretrained(
        model,
        args.model_path,
        is_trainable=False
    )
    
    model = model.to(torch.bfloat16)
    model.eval()
    
    # dataset = RadDataset(split='test')
    # dataset = SlakeDataset(split='test')
    # dataset = OvqaDataset(split='test')
    dataset = get_dataset_constructor(args.dataset)(split='test')
    
    ans_file = open(args.output_path, "w")
    
    tokenizer.add_tokens([AddedToken('<image>', single_word=False, lstrip=True, rstrip=True, special=True)])
    image_token_id = tokenizer.convert_tokens_to_ids('<image>')
    
    for line in tqdm(dataset):
        qid, question, answer, image = line

        question = question.replace(DEFAULT_IMAGE_TOKEN, '').strip()
        cur_prompt = question
        question = DEFAULT_IMAGE_TOKEN + '\n' + question
        
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        image_tensor = process_images([image], image_processor, model.config)[0]

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda().to(torch.bfloat16),
                max_new_tokens=1024,
                use_cache=False,
            )

        output_ids = output_ids.clone()
        input_length = input_ids.size(1)
        output_ids = output_ids[:, input_length:]
        
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_file.write(
            json.dumps({
                "qid": qid, 
                "question": cur_prompt, 
                "response": outputs,
                'label': answer,
            })
            + "\n"
        )
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conv-mode", type=str, default="mistral_instruct")
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--output-path", type=str)
    args = parser.parse_args()

    eval_model(args)
