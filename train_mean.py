import json
from os.path import join,isfile

from transformers import AutoTokenizer,AutoModel
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader

from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

import random 

from tqdm import tqdm
#import numpy as np

def get_hebrew_val():
	with open(join('data','tdklab___hebrew_squad_v1','validation.json')) as f:
		return json.load(f)['data']

def get_english_val():
    from datasets import load_dataset
    return load_dataset("squad",split='validation')

def get_hebrew_train():
    with open(join('data','tdklab___hebrew_squad_v1','train.json')) as f:
        return json.load(f)['data']

def get_english_train():
    from datasets import load_dataset
    return load_dataset("squad",split='train')

def deranged_shuffle(lst, seed=None):
    lst=list(lst)
    assert isinstance(lst,list)
    if seed is not None:
        random.seed(seed)

    n = len(lst)
    shuffled = lst[:]
    random.shuffle(shuffled)

    # Identify positions where elements are in their original place
    same_position = [i for i, (a, b) in enumerate(zip(lst, shuffled)) if a == b]

    for i in same_position:
        swap_with = random.choice([k for k in range(n) if k != i and shuffled[k] != lst[k] and k not in same_position])
        shuffled[i], shuffled[swap_with] = shuffled[swap_with], shuffled[i]

    return shuffled

class Retrival_Data(Dataset):
    def __init__(self,data,seed=None):
        self.data=data#['question']
        self.wrong=deranged_shuffle(data,seed)
        self.wrong=[x['answers']['text'][0] for x in self.wrong]
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        q=self.data[idx]['question']
        a=self.data[idx]['answers']['text'][0]
        w=self.wrong[idx]
        return q,a,w

    def reshufle(self):
        self.wrong=deranged_shuffle(self.data)
        self.wrong=[x['answers']['text'][0] for x in self.wrong]


def get_batcher(tokenizer):
    def batcher(data):
        q,a,w=[tokenizer([x[i] for x in data],return_tensors='pt',padding=True) for i in range(3)]
        return q,a,w

    return batcher
        

def train_loop(model, data, optimizer=None,scedualer=None, margin=1.):
    dis = lambda x,y: torch.norm(x-y,dim=1)#F.cosine_similarity
    running_loss = 0.0
    non_zeroed_losses = 0
    correct_predictions = 0
    num_batches = 0

    captions = 'Training' if optimizer else "Testing"
    # Initialize tqdm progress bar
    progress_bar = tqdm(data, desc=captions)

    for b in progress_bar:
        b = [{k: v.to(model.device) for k, v in d.items()} for d in b]
        b = [model(**d).pooler_output for d in b]
        (q, a, w) = b

        # Compute cosine similarities
        dis_q_a = dis(q, a)
        dis_q_w = dis(q, w)


        # Compute the raw loss before clamping
        raw_loss = (dis_q_a - dis_q_w) + margin

        # Count non-zero losses
        non_zeroed_losses += (raw_loss >= 0).detach().sum().cpu().item()

        # Count correct predictions (accuracy)
        correct_predictions += (dis_q_w > dis_q_a).detach().sum().cpu().item()

        # Apply clamping and compute mean loss for the batch
        loss = raw_loss.clamp(min=0).mean()

        # Update running loss and number of elements processed
        running_loss += loss.detach().cpu().item() * q.shape[0]
        num_batches += q.shape[0]

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Update progress bar with current average loss, proportion of non-zero losses, and accuracy
        progress_bar.set_postfix(avg_loss=running_loss / num_batches,
                                 non_zeroed_pct=100.0 * non_zeroed_losses / num_batches,
                                 accuracy=100.0 * correct_predictions / num_batches)
    
    return running_loss / num_batches, 100.0 * correct_predictions / num_batches,100.0*non_zeroed_losses / num_batches #non_zeroed_losses / num_batches

def save_model_with_unique_name(model,tokenizer, model_name, directory='models'):
    version = 0
    model_dir_name = f"{model_name}_v{version}"
    model_path = join(directory, model_dir_name)

    # Check for existing model directories and increment version number
    while isfile(join(model_path, 'config.json')) or isfile(join(model_path, 'pytorch_model.bin')):
        version += 1
        model_dir_name = f"{model_name}_v{version}"
        model_path = join(directory, model_dir_name)

    # Save the model using save_pretrained
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
    print(f"Model saved in directory {model_path}")

    return model_path

if __name__=="__main__":
    #trying stuff from https://aclanthology.org/D19-1410.pdf
    #model_name="bert-base-multilingual-cased"
    #model_name="avichr/Legal-heBERT"
    #model_name="avichr/heBERT"
    model_name="bert-base-uncased"

    model=AutoModel.from_pretrained(model_name)
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    model.to('cuda')

    #print(tokenizer(['helo world'],return_tensors='pt')).keys()
    #print(model.pooler.dense.weight)
    #print(model(**tokenizer(['helo world'],return_tensors='pt')).keys())

    train=get_english_train()
    test=get_english_val()

    #print(np.mean([len(x['answers']) for x in test]))

    train=Retrival_Data(train)
    test=Retrival_Data(test,seed=42)
    #print(train[0])
    
    train_loader=DataLoader(train,batch_size=32,collate_fn=get_batcher(tokenizer))
    test_loader=DataLoader(test,batch_size=128,collate_fn=get_batcher(tokenizer),shuffle=False)

    opt=Adam(model.parameters(),lr=2e-5)#0.00001)
    # Learning Rate Scheduler setup
    def slanted_triangular_lr(iteration):
        if iteration < cut_frac * total_iterations:
            return iteration / (cut_frac * total_iterations)
        return 1 - (iteration - cut_frac * total_iterations) / ((1 - cut_frac) * total_iterations)

    num_epochs = 2
    total_iterations = num_epochs * len(train_loader)  # Adjust the data_loader for your dataset
    cut_frac = 0.1
    scheduler = LambdaLR(opt, lr_lambda=slanted_triangular_lr)
    
    margin=1. 

    for i in range(num_epochs):
        train.reshufle()
        train_loop(model,train_loader,opt,scheduler,margin)
        train_loop(model,test_loader,None)


    save_model_with_unique_name(model,tokenizer, model_name)