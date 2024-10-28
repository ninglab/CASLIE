import requests
from PIL import Image
from tqdm import tqdm
import torch
import os
from transformers import Blip2Processor, Blip2ForConditionalGeneration, AutoConfig, AutoProcessor, LlavaForConditionalGeneration, LlavaNextProcessor, LlavaNextForConditionalGeneration, MllamaForConditionalGeneration
import argparse
from datasets import load_dataset
import pandas as pd

from template import *

def get_captions(data, spl, task, model_name='llama'):
    fp = f'captions/{task}_{spl}.json'
    if os.path.exists(fp):
        return json.load(open(fp, 'r'))
    if not os.path.exists('captions'):
        os.mkdir('captions')

    if model_name == 'llava':
        captions = caption_llava(data)
    elif model_name == 'llava-next':
        captions = caption_llavanext(data)
    elif model_name == 'blip2':
        captions = caption_blip2(data)
    else:
        captions = caption_llama(data)
    with open(fp, 'w') as f:
        json.dump(captions, f)
    
    return captions

def caption_blip2(data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_conf = AutoConfig.from_pretrained('Salesforce/blip2-opt-2.7b', max_new_tokens=128)
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16, config=text_conf)
    
    def generate_caption(image_url):
        try:
            image = Image.open(requests.get(image_url, stream=True).raw)
        except:
            return 'NA'
        inputs = processor(images=image, return_tensors="pt").to(device=device, dtype=torch.float16)
        generated_ids = model.generate(**inputs)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].replace('\n', ' ').replace('\t', ' ').strip()
        
        if len(generated_text) < 2:
            generated_text = 'NA'
        return generated_text

    captions = []
    for entry in data:
        generated_caps = []
        for img in json.loads(entry['images']):
            generated_caps.append(generate_caption(img))
        captions.append(generated_caps)

    return captions

def caption_llava(data):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf").to(device)
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", device=device)
    processor.tokenizer.padding_side = "left"

    def generate_caption(prpt, image_url):
        try:
            image = Image.open(requests.get(image_url, stream=True).raw)
        except:
            return 'NA'
        prompt = "USER: <image>\nPlease generate an informative caption for the product in the image. {} ASSISTANT:".format(prpt)
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

        generate_ids = model.generate(**inputs, max_new_tokens=128)
        generated_text = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        generated_text = str(generated_text).split('ASSISTANT: ')[-1].replace('\n', ' ').replace('\t', ' ').strip()
        if len(generated_text) < 2:
            generated_text = 'NA'
        return generated_text

    captions = []
    for entry in data:
        generated_caps = []
        prompts = caption_prompt(entry)
        for i, img in enumerate(json.loads(entry['images'])):
            generated_caps.append(generate_caption(prompts[i], img))
        captions.append(generated_caps)

    return captions

def caption_llavanext(data):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)

    def generate_caption(prpt, image_url):
        try:
            image = Image.open(requests.get(image_url, stream=True).raw)
        except:
            return 'NA'
        conversation = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Please generate a concise and informative caption for the product in the image. {}".format(prpt)},],
        },]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")

        output = model.generate(**inputs, max_new_tokens=128)
        generated_text = processor.decode(output[0], skip_special_tokens=True)
        generated_text = str(generated_text).split('[/INST]')[-1].replace('\n', ' ').replace('\t', ' ').strip().strip('"')
        if len(generated_text) < 2:
            generated_text = 'NA'

        return generated_text

    captions = []
    for entry in data:
        generated_caps = []
        prompts = caption_prompt(entry)
        for i, img in enumerate(json.loads(entry['images'])):
            generated_caps.append(generate_caption(prompts[i], img))
        captions.append(generated_caps)

    return captions

def caption_llama(data):

    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)

    def generate_caption(prpt, image_url):
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": 'Please generate a concise and informative caption for the product in the image within 50 words. {}'.format(prpt)}
            ]}
        ]
        image = Image.open(requests.get(image_url, stream=True).raw)
        input_text = processor.apply_chat_template(messages,add_generation_prompt=True)
        inputs = processor(
            images=image,
            text=input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=128)
        generated_text = processor.decode(outputs[0])
        generated_text = generated_text.split('<|end_header_id|>')[-1].split('<|eot_id|>')[0].replace('\n', ' ').replace('\t', ' ').strip()
        if len(generated_text) < 2:
            generated_text = 'NA'
        return generated_text

    captions = []
    for entry in tqdm(data):
        generated_caps = []
        prompts = caption_prompt(entry)
        for i, img in enumerate(json.loads(entry['images'])):
            generated_caps.append(generate_caption(prompts[i], img))
        captions.append(generated_caps)

    return captions    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--task', default='answerability_prediction')
    parser.add_argument('-spl','--split', default='test')
    parser.add_argument('-cm','--captioning_model', default='llama')
    args = parser.parse_args()
    print(args)

    df_data = pd.DataFrame(load_dataset("NingLab/MMECInstruct")['train'])
    df_data = df_data[(df_data['task'] == args.task) & (df_data['split'] == args.split)]
    data = df_data.to_dict(orient='records')
    get_captions(data, args.split, args.task, args.captioning_model)