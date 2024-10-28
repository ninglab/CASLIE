from captioning import *
import json
import os
import pandas as pd
from collections import Counter
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from datasets import load_dataset
import pdb

def eval_captioins(spl, task, is_gen_caps):
    df_data = pd.DataFrame(load_dataset("NingLab/MMECInstruct")['train'])
    df_data = df_data[(df_data['task'] == task) & (df_data['split'] == spl)]
    data = df_data.to_dict(orient='records')
    if is_gen_caps:
        captions = get_captions(data, spl, task, model_name='llama')
    else:
        captions = None

    preds_all = {}
    predict_model_set = ['llama-3b', 'llama-8b', 'llama-vl', 'mistral', 'phi']
    for predict_model in predict_model_set:
        preds_all[predict_model] = predict_captions(data, captions, predict_model, is_gen_caps)

    votes = []
    for i, entry in enumerate(data):
        vote = []
        for j in range(len(json.loads(entry['images']))):
            votes_mv = [preds_all[m][i][j] for m in predict_model_set]
            counter = Counter(votes_mv)
            vote.append(counter.most_common(1)[0][0])
        votes.append(vote)
    
    return votes


def predict_captions(data, captions, predict_model, is_gen_caps):

    if predict_model == 'phi':
        model_id = 'microsoft/Phi-3.5-mini-instruct'
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map="cuda", 
            torch_dtype="auto", 
            trust_remote_code=True, 
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )
    elif predict_model == 'llama-vl':
        model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(model_id)
    else:
        if predict_model == 'mistral':
            model_id = "mistralai/Mistral-7B-Instruct-v0.3"
        elif predict_model == 'llama-3b':
            model_id = "meta-llama/Llama-3.2-3B-Instruct"
        else:
            model_id = "meta-llama/Llama-3.1-8B-Instruct"

        pipe = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto"
        )

    def predict_item(prompt):
        if predict_model == 'llama-vl':
            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": prompt}
                ]}
            ]
            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(
                images=None,
                text=input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(model.device)

            outputs = model.generate(**inputs, max_new_tokens=16)
            generated_text = processor.decode(outputs[0]).split('\n')[-1].split('<')[0].strip()[:3]
        elif predict_model == 'phi':
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt},
            ]
            generation_args = {
                "max_new_tokens": 16,
                "return_full_text": False,
                "do_sample": False,
            }

            output = pipe(messages, **generation_args)
            generated_text = output[0]['generated_text'].replace('\n', ' ').strip()[:3]
        else:
            messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt},
            ]

            outputs = pipe(
                messages,
                max_new_tokens=16,
                pad_token_id=None,
                eos_token_id=None
            )
            generated_text = outputs[0]["generated_text"][-1]['content'].replace('\n', ' ').strip()[:3]
        generated_text = generated_text.strip().strip('.').strip(',').lower()

        return generated_text

    predictions = []
    for entry in tqdm(data):
        generated_preds = []
        prompts = pred_prompt(entry, is_gen_caps, captions)
        for prompt in prompts:
            generated_preds.append(predict_item(prompt))
        predictions.append(generated_preds)

    return predictions

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--task', default='answerability_prediction')
    parser.add_argument('-spl','--split', default='test')
    parser.add_argument('-is_cap','--is_gen_caps', default=False)
    args = parser.parse_args()
    print(args)
    
    eval_captioins(args.spl, args.task, args.is_gen_caps)
