# CASLIE

The repo contains the code for [Captions Speak Louder than Images (CASLIE): Generalizing Foundation Models for E-commerce from High-quality Multimodal Instruction Data](https://arxiv.org/abs/2410.17337).

## Introduction
We introduce [MMECInstruct](https://huggingface.co/datasets/NingLab/MMECInstruct), the first-ever, large-scale, and high-quality multimodal instruction dataset for e-commerce. 
We also develop [CASLIE](https://huggingface.co/NingLab/CASLIE-M), a simple, lightweight, yet effective framework for integrating multimodal information.
Within CASLIE, we develop a series of state-of-the-art multimodal foundation models for e-commerce by instruction tuning using MMECInstruct.

## Requirements

* python = 3.10.14
* torch = 2.4.1
* transformers = 4.46.0
* fire = 0.7.0
* scikit-learn = 1.5.2
* datasets = 3.0.1

## MMECInstruct Dataset

The dataset is available in [Hugging Face](https://huggingface.co/datasets/NingLab/MMECInstruct).
MMECInstruct comprises 7 tasks, including
answerability prediction, category classification, product relation prediction, 
product substitute identification, multiclass product classification, 
sentiment analysis, and sequential recommendation. 
MMECInstruct is split into training sets, validation sets, IND
test sets, and OOD test sets.


## Enriched Context-conditioned Captioning
To generate textual captions for products, run <code>python captioning.py --task $task --split $split --captioning_model $captioning_model</code>.
 <!-- By default we use [Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct) as the captioning model. -->

<code>$task</code> specifies the data of which task to be captioned.

<code>$split</code> specifies the data of which split to be captioned.

<code>$captioning_model</code> specifies the captioning model to be used, choosing from [llama](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct), [llava](https://huggingface.co/llava-hf/llava-1.5-7b-hf), [llava-next](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf), and [blip2](https://huggingface.co/Salesforce/blip2-opt-2.7b).

Example:
```
python captioning.py --task answerability_prediction --split test --captioning_model llama
```

## Caption Quality Evaluation
To evaluate the quality of generated captions, run <code>python cqe.py --task $task --split $split --is_gen_caps $is_gen_caps</code>.

<code>$task</code> specifies the captions of which task to be evaluated.

<code>$split</code> specifies the captions of which split to be evaluated.

<code>$is_gen_caps</code> specifies whether to use generated captions or not (use the captions provided in MMECInstruct).

Example:
```
python cqe.py --task answerability_prediction --split test --is_gen_caps True
```

If you set <code>$is_gen_caps</code> as True but do not generate the captions before, the script will generate the captions first and then evaluate the quality.

##  Modality-unified Inference
To conduct inference, run <code>python inference.py --model_path $model_path --task $task --output_path $output_path</code>.

<code>$model_path</code> is the path of the instruction-tuned model.

<code>$task</code> specifies the task to be tested.

<code>$output_path</code> specifies the path where you want to save the inference output.

Example:
```
python inference.py --model_path NingLab/CASLIE-M --task answerability_prediction --output_path ap.json
```

## Evaluation
To evaluate the model on specific tasks, run <code>python evaluate.py --task $task --output_path $output_path</code>.

<code>$task</code> is the task on which you want to conduct the evaluation.

<code>$output_path</code> specifies where the inferenced result store in.

Example:
```
python evaluate.py --task answerability_prediction --output_path ap.json
```


## Citation
```bibtex
@article{ling2024captions,
    title={Captions Speak Louder than Images (CASLIE): Generalizing Foundation Models for E-commerce from High-quality Multimodal Instruction Data},
    author={Ling, Xinyi and Peng, Bo and Du, Hanwen and Zhu, Zhihui and Ning, Xia},
    journal={arXiv preprint arXiv:2410.17337},
    year={2024}
}
```