import json
import pdb

def caption_prompt(entry):
    task = ''.join([s[0] for s in entry['task'].split('_')])
    inp = json.loads(entry['input'])
    if task == 'mpc':
        prompts = ["The caption should be helpful to predict the relevance between the user's query: '{}', and product: '{}'.".format(inp['query'], inp['product title'])]
    elif task == 'psi':
        prompts = ["The caption should be helpful to predict if the product: '{}' can serve as a functional substitute for the user's query: '{}'.".format(inp['query'], inp['product title'])]
    elif task == 'prp':
        prompts = ["The title of the product in the image is '{}'. The caption should be helpful to predict the relation between this product and the other product '{}'.".format(inp[f'product 1'], inp[f'product 2']), "The title of the product in the image is '{}'. The caption should be helpful to predict the relation between this product and the other product '{}'.".format(inp[f'product 2'], inp[f'product 1'])]
    elif task == 'sa':
        prompts = ["The caption should be helpful to identify the user's sentiment from the review: {}.".format(inp['review'])]
    elif task == 'ap':
        prompts = ["The caption should be helpful to identify if the product-related question: {}, is answerable.".format(inp['question'])]
    elif task == 'sr':
        prompts = []
        for i in range(len(inp)):
            prompts.append("The caption should be helpful to recommend the next product the user is most likely to purchase by analyzing the user's intent based on the user's purchase history. Here is the title of the profuct in the image: {}.".format(inp[i].split(': ', 1)[-1]))
        ops = json.loads(entry['options'])
        for i in range(len(ops)):
            prompts.append("The caption should be helpful to recommend the next product the user is most likely to purchase by analyzing the user's intent based on the user's purchase history. Here is the title of the profuct in the image: {}.".format(ops[i].split(': ', 1)[-1]))
    elif task == 'cc':
        prompts = ["The caption should be helpful to identify the product's fine-grained category. Here is the product title: {}.".format(inp['title'])]
    return prompts

    
def pred_prompt(entry, is_gen_caps=False, captions=None):
    task = ''.join([s[0] for s in entry['task'].split('_')])
    if not is_gen_caps:
        captions = json.loads(entry['caption_info'])
    inp = json.loads(entry['input'])
    if task == 'mpc':
        prompts = [f"The model needs to predict the relevance between the query and product by analyzing the user's query: '{inp['query']}', and product title: '{inp['product title']}'. Here is the additional information about the product extracted from product image: '{captions[0]}', you need to determine if the information extracted from the image will be helpful in predicting the relevance result. Only output yes or no."]
    elif task == 'psi':
        prompts = [f"The model needs to identify if the product is somewhat relevant to the query but fails to fulfill some aspects of the query but the product can be used as a functional substitute. Given a user's query: '{inp['query']}' and a product title: '{inp['product title']}', as well as additional information about the product extracted from the product image: '{captions[0]}', you need to determine if the information extracted from the image will be helpful in identifying the relevance between the product and the query. Only output yes or no."]
    elif task == 'sa':
        prompts = [f"The task needs to identify the user's sentiment based on their review: '{inp['review']}'. Here is the additional information about the product extracted from the user review's image: '{captions[0]}'. You need to determine if the information extracted from the image will help to identify the user's sentiment. Only output yes or no."]
    elif task == 'ap':
        prompts = ["The task needs to identify if the question: '{}', is answerable based on the related document. Here is the additional information about the product that extracted from the product image: '{}'. You need to determine if the information extracted from the image will help to identify the question's answerability. Only output yes or no.".format(inp['question'], captions[0])]
    elif task == 'cc':
        prompts = ["The task needs to identify the product's fine-grained category. Here is the product title : '{}', and additional information about the product that extracted from the product image: '{}'. You need to determine if the information extracted from the image will help to identify the category. Only output yes or no.".format(inp['title'], captions[0])]
    elif task == 'sr':
        prompts = []
        for i in range(len(inp)):
            prompts.append("The task needs to recommend the next product that the user may be interested in based on the user's purchase history. Here is the title of a product from purchase history: '{}', and the information extracted from the product image: {}. You need to determine if the information extracted from the image will be helpful for recommendation. Only output yes or no.".format(inp[i].split(': ', 1)[-1], captions[i]))
        ops = json.loads(entry['options'])
        for i in range(len(ops)):
            prompts.append("The task needs to recommend the next product that the user may be interested in based on the user's purchase history. Here is the title of a product from purchase history: '{}', and the information extracted from the product image: {}. You need to determine if the information extracted from the image will be helpful for recommendation. Only output yes or no.".format(ops[i].split(': ', 1)[-1], captions[i+len(inp)]))
    elif task == 'prp':
        p1 = "The model needs to identify if the two products are similar or will be purchased together or be viewed together given the title of the product 1: '{}', and the product 2: '{}'. Here is the additional information about the product 1 extracted from its image: '{}', you need to determine if the information extracted from the image will be helpful in identifying the relation between the two products. Only output yes or no.".format(inp['product 1'], inp['product 2'], captions[0])
        p2 = "The model needs to identify if the two products are similar or will be purchased together or be viewed together given the title of the product 1: '{}', and the product 2: '{}'. Here is the additional information about the product 2 extracted from its image: '{}', you need to determine if the information extracted from the image will be helpful in identifying the relation between the two products. Only output yes or no.".format(inp['product 1'], inp['product 2'], captions[1])
        prompts = [p1, p2]

    return prompts