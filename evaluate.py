import fire
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
import json
import pandas as pd
import pdb

import warnings
warnings.filterwarnings("ignore")

def cpt_bicls(data, prediction_list, is_print):
    label_list = [entry['output'] for entry in data]

    skipped = 0
    filtered_prediction_list = []
    filtered_label_list = []
    valid_response = set(label_list)
    for i in range(len(prediction_list)):
        pred = prediction_list[i].strip()
        if ':' in pred:
            pred = pred[0]
        if pred in valid_response:
            filtered_prediction_list.append(pred)
            filtered_label_list.append(label_list[i])
        else:
            skipped += 1
            
    acc = accuracy_score(filtered_label_list, filtered_prediction_list)
    pre = precision_score(filtered_label_list, filtered_prediction_list, pos_label = 'yes')
    rec = recall_score(filtered_label_list, filtered_prediction_list, pos_label = 'yes')
    f1 = f1_score(filtered_label_list, filtered_prediction_list, pos_label = 'yes')

    if is_print:
        print(f'& {acc:.3f} & {pre:.3f} & {rec:.3f} & {f1:.3f} & {skipped}')

    return acc, pre, rec, f1, skipped

def cpt_mcls(data, prediction_list, is_print):
    label_list = [entry['output'] for entry in data]

    skipped = 0
    filtered_prediction_list = []
    filtered_label_list = []
    valid_response = set(label_list)
    for i in range(len(prediction_list)):
        pred = prediction_list[i].strip()
        if ':' in pred:
            pred = pred[0]
        if pred in valid_response:
            filtered_prediction_list.append(pred)
            filtered_label_list.append(label_list[i])
        else:
            skipped += 1
            
    acc = accuracy_score(filtered_label_list, filtered_prediction_list)
    pre_macro = precision_score(filtered_label_list, filtered_prediction_list, average = 'macro')
    rec_macro = recall_score(filtered_label_list, filtered_prediction_list, average = 'macro')
    f1_macro  = f1_score(filtered_label_list, filtered_prediction_list, average = 'macro')

    if is_print:
        print(f'& {acc:.3f} & {pre_macro:.3f} & {rec_macro:.3f} & {f1_macro:.3f} & {skipped}')

    return acc, pre_macro, rec_macro, f1_macro, skipped

def cpt_rec1(data, prediction_list, is_print):
    label_list = [entry['output'] for entry in data]

    skipped = 0
    filtered_prediction_list = []
    filtered_label_list = []
    valid_response = set(label_list)

    for i in range(len(prediction_list)):
        pred = prediction_list[i].strip()
        # pdb.set_trace()
        if ':' in pred:
            pred = pred[0]
        if pred in valid_response:
            filtered_prediction_list.append(pred)
            filtered_label_list.append(label_list[i])
        else:
            skipped += 1
    
    acc = accuracy_score(filtered_label_list, filtered_prediction_list)
    if is_print:
        print(f'& {acc:.3f} & {skipped}')

    return acc, skipped

def eval(
        task,
        output_path,
        is_print = True
):
    df_data = pd.DataFrame(json.load(open('../predictor/data_hf/dataset.json', 'r')))
    test = df_data[(df_data['task'] == task) & (df_data['split'] == 'test')]
    test = test.to_dict(orient='records')
    prediction_list = json.load(open(output_path, 'r'))

    if task in ['ap', 'ori_psi']:
        acc, pre, rec, f1, skipped = cpt_bicls(test, prediction_list, is_print)
    elif task in ['cc', 'sr']:
        acc, skipped = cpt_rec1(test, prediction_list, is_print)
    elif task == 'mpc':
        acc, pre, rec, f1, skipped = cpt_mcls(test, prediction_list, is_print)
    elif task in ['prp', 'sa']:
        acc, pre, rec, f1, skipped = cpt_mcls(test, prediction_list, is_print)
    
if __name__ == '__main__':
    fire.Fire(eval)