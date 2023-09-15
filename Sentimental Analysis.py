Sentimental_analysis.py
import json
import re
import pandas as pd
from sklearn import preprocessing
import torch
import requests
import numpy as np
import os
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from tqdm.notebook import tqdm
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from keras.utils import pad_sequences
 
# deal = input("Enter the deal")
# investor = input("Enter the investor associated with the deal")
# requests.get('https://huggingface.co/', verify=False)
df = pd.read_csv(r'filename', sep=',', engine='python', quotechar='"', error_bad_lines=False)
 
df = df.dropna()
class Config():
    seed_val = 17
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 5
    batch_size = 6
    seq_length = 512
    lr = 2e-5
    eps = 1e-8
    pretrained_model = 'bert-base-uncased'
    test_size=0.15
    random_state=42
    add_special_tokens=True
    return_attention_mask=True
    pad_to_max_length=True
    do_lower_case=False
    return_tensors='pt'
 
config = Config()
 
# params will be saved after training
params = {"seed_val": config.seed_val,
    "device":str(config.device),
    "epochs":config.epochs,
    "batch_size":config.batch_size,
    "seq_length":config.seq_length,
    "lr":config.lr,
    "eps":config.eps,
    "pretrained_model": config.pretrained_model,
    "test_size":config.test_size,
    "random_state":config.random_state,
    "add_special_tokens":config.add_special_tokens,
    "return_attention_mask":config.return_attention_mask,
    "pad_to_max_length":config.pad_to_max_length,
    "do_lower_case":config.do_lower_case,
    "return_tensors":config.return_tensors,
        }
 
# set random seed and device
device = config.device
 
 
random.seed(config.seed_val)
np.random.seed(config.seed_val)
torch.manual_seed(config.seed_val)
torch.cuda.manual_seed_all(config.seed_val)
 

train_df_, val_df = train_test_split(df,
                                    test_size=0.10,
                                    random_state=config.random_state, stratify=df.label.values)
 
train_df, test_df = train_test_split(train_df_,
                                    test_size=0.10,
                                    random_state=42, stratify=train_df_.label.values)
 
tokenizer = BertTokenizer.from_pretrained(config.pretrained_model,
                                          do_lower_case=config.do_lower_case)
 

encoded_data_train = tokenizer.batch_encode_plus(
    train_df.cleanse_data.values,
    truncation =  True,
    add_special_tokens=config.add_special_tokens,
    return_attention_mask=config.return_attention_mask,
    pad_to_max_length=config.pad_to_max_length,
    max_length=config.seq_length,
    return_tensors=config.return_tensors
)
encoded_data_val = tokenizer.batch_encode_plus(
    val_df.cleanse_data.values,
    truncation =  True,
    add_special_tokens=config.add_special_tokens,
    return_attention_mask=config.return_attention_mask,
    pad_to_max_length=config.pad_to_max_length,
    max_length=config.seq_length,
    return_tensors=config.return_tensors
)
 
# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(train_df.cleanse_data)
# train_array = X.toarray()
# Y = vectorizer.fit_transform(val_df.cleanse_data)
# val_array = Y.toarray()
 
input_ids_train =  encoded_data_train['input_ids']
attention_masks_train =  encoded_data_train['attention_mask']
le = preprocessing.LabelEncoder()
targets_train = le.fit_transform(train_df.label.values)
labels_train = torch.tensor(targets_train).type(torch.LongTensor)
 
# labels_train = labels_train - 1
# print(len(torch.unique(labels_train))) # calculate the number classes in this tensor
# Define the batch size
# batch_size = 32
 
# # Calculate the number of batches
# num_batches = len(train_array) // batch_size
 
# # Loop over the number of batches
# for i in range(num_batches):
#     # Get the current batch of data
#     batch_data = train_array[i * batch_size:(i + 1) * batch_size]
   
#     # Convert the data to a tensor
#     if i == 0:    
#         labels_train = torch.tensor(batch_data)
#     else:
#         labels_train = torch.cat((labels_train, torch.tensor(batch_data)), dim =0)
                           
# labels_train = torch.tensor(train_array)
 

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
targets_val = le.fit_transform(val_df.label.values)
labels_val = torch.tensor(targets_val).type(torch.LongTensor)
# targets_val = le.fit_transform(val_df.cleanse_data.values)
# labels_val = torch.tensor(targets_val)c
# labels_val = labels_val.type(torch.LongTensor)
 

# val_batches = len(val_array) // batch_size
 
# # Loop over the number of batches
# for i in range(val_batches):
#     # Get the current batch of data
#     batch_data = val_array[i * batch_size:(i + 1) * batch_size]
   
#     # Convert the data to a tensor
#     if i == 0:    
#         labels_val = torch.tensor(batch_data)
#     else:
#         labels_val = torch.cat((labels_val, torch.tensor(batch_data)), dim =0)
   
# labels_val = torch.tensor(test_array)
dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)#, labels_train) #making a list of tensors
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)#, labels_val)
 
#create the model
model = BertForSequenceClassification.from_pretrained(config.pretrained_model,
                                                      num_labels=3,
                                                      output_attentions=False,
                                                      output_hidden_states=False)
 
dataloader_train = DataLoader(dataset_train,
                              sampler=RandomSampler(dataset_train),
                              batch_size=config.batch_size)
 
dataloader_validation = DataLoader(dataset_val,
                                   sampler=SequentialSampler(dataset_val),
                                   batch_size=config.batch_size)                                                    
optimizer = AdamW(model.parameters(),
                  lr=config.lr,
                  eps=config.eps)
                 
 
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train)*config.epochs)
 
def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')
 
def accuracy_per_class(preds, labels, label_dict):
    label_dict_inverse = {v: k for k, v in label_dict.items()}
   
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
 
    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')
 
#Training Loop
def evaluate(dataloader_val):
 
    model.eval()
   
    loss_val_total = 0
    predictions, true_vals = [], []
   
    for batch in dataloader_val:
        batch = tuple(b.to(config.device) for b in batch)
       
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }
 

        with torch.no_grad():        
            outputs = model(**inputs)
 
        loss = outputs.loss
        logits = outputs.logits
        if(loss is not None):
            loss_val_total += loss.item()
 
        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
       
    # calculate avareage val loss
    loss_val_avg = loss_val_total/len(dataloader_val)
   
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
           
    return loss_val_avg, predictions, true_vals                                          
 
model.to(config.device)
   
for epoch in tqdm(range(1, config.epochs+1)):
   
    model.train()
   
    loss_train_total = 0
    # allows you to see the progress of the training
    progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
   
    i = 0
    for batch in progress_bar:
        model.zero_grad()
       
        batch = tuple(b.to(config.device) for b in batch)
       
       
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }  
             
   
        outputs = model(**inputs)
        loss = outputs[0]
        loss_tuple = (loss,)
        loss_train_total = torch.stack(loss_tuple)
        # loss_train_total += loss.item()
        loss.retain_grad()
        loss.sum().backward()
 
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
 
        optimizer.step()
        scheduler.step()
       
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.sum().item()/len(batch))})
       
    torch.save(model.state_dict(), f'_BERT_epoch_{epoch}.model')
       
    tqdm.write(f'\nEpoch {epoch}')
   
    loss_train_avg = loss_train_total/len(dataloader_train)            
    tqdm.write(f'Training loss: {loss_train_avg}')
   
    val_loss, predictions, true_vals = evaluate(dataloader_validation)
    val_f1 = f1_score_func(predictions, true_vals)
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'F1 Score (Weighted): {val_f1}');
# save model params and other configs
with Path('params.json').open("w") as f:
    json.dump(params, f, ensure_ascii=False, indent=4)
 
model.load_state_dict(torch.load(f'./_BERT_epoch_3.model', map_location=torch.device('cpu')))
 
preds_flat = np.argmax(predictions, axis=1).flatten()
print(classification_report(preds_flat, true_vals))
 
 
 
