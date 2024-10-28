from captioning import *
from cqe import *
import json
import os
import pandas as pd
from collections import Counter
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from datasets import load_dataset
import pdb

def get_data(task, spl, is_gen_caps=False):
    if os.path.exists(f'data/{task}_{spl}.json'):
        return json.load(open(f'data/{task}_{spl}.json', 'r'))
    df_data = pd.DataFrame(load_dataset("NingLab/MMECInstruct")['train'])
    df_data = df_data[(df_data['task'] == task) & (df_data['split'] == spl)]
    data = df_data.to_dict(orient='records')
    if is_gen_caps:
        captions = get_captions(data, spl, task, model_name='llama')
    else:
        captions = None

    votes = eval_captioins(spl, task, is_gen_caps)
    new_data = []
    for i, entry in enumerate(data):
        caption = captions[i] if is_gen_caps else json.loads(entry['caption_info'])
        vote = votes[i]
        new_entry = {'instruction': entry['instruction']}
        ops = json.loads(entry['options'])
        inp = json.loads(entry['input'])
        if task == 'sequential_recommendation':
            for j in range(len(inp)):
                inp[j] = inp[j] + ' ' + caption[j] if vote[j] == 'yes' else inp[j]
            for j in range(len(ops)):
                ops[j] = ops[j] + ' ' + caption[j+len(inp)] if vote[j+len(inp)] == 'yes' else ops[j]
        elif task == 'product_relation_prediction':
            inp['product 1 information extracted from image'] = caption[0] if vote[0] == 'yes' else None
            inp['product 2 information extracted from image'] = caption[1] if vote[1] == 'yes' else None
        else:
            inp['product information extracted from image'] = caption[0] if vote[0] == 'yes' else None
        new_entry['input'] = json.dumps(inp)
        new_entry['options'] = json.dumps(ops)
        new_data.append(new_entry)
    json.dump(open(f'data/{task}_{spl}.json', 'w'))
    return new_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--task', default='answerability_prediction')
    parser.add_argument('-spl','--split', default='test')
    parser.add_argument('-is_cap','--is_gen_caps', default=False)
    args = parser.parse_args()
    print(args)

    get_data(args.task, args.split, args.is_gen_caps)
