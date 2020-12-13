# -*- coding: utf-8 -*-
"""
# Part 1 : Sentiment Classification

Data : https://drive.google.com/file/d/1JQDD4e16uyN2gXKfcEfeKUCsaRrgrYM1/view

## Import the package and dataset
"""

!python --version
!pip freeze | grep torch
from google.colab import drive
!pip install transformers
drive.mount('/content/drive')
!unzip -qq ./drive/My\ Drive/twitter_sentiment.zip

"""## Data preprocessing"""

# a = tokenizer.encode_plus("Hey! How are you today?")
# print(a['attention_mask'])

import csv
import os
import numpy as np
import torch
import re
from transformers import BertTokenizer
import torch
from transformers import BertTokenizer, BertModel
from transformers import AlbertTokenizer, AlbertForMaskedLM
from torch.nn.utils.rnn import pad_sequence
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class textData(torch.utils.data.Dataset):
    def __init__(self, csv_file, mode='train', transform=None):
        self.mode = mode # 'train', 'val' or 'test'
        self.data_list = []
        self.labels = []
        self.input_ids = []
        input_ids = []
        self.attention_masks = []    
        self.tokenizer = tokenizer           
        with open(csv_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)                                  
            for row in reader:
              row['text'] = row['text'].lower()
              # Remove @name
              row['text'] = re.sub(r'(@.*?)[\s]', ' ', row['text'])
              # Replace '&amp;' with '&'
              row['text'] = re.sub(r'&amp;', '&', row['text'])
              # Remove trailing whitespace
              row['text'] = re.sub(r'\s+', ' ', row['text']).strip()
              row['text'] = tokenizer.tokenize(row['text'])
              id =tokenizer.encode(row['text'], add_special_tokens=True)
              mask = tokenizer.encode_plus(row['text'])             
              attention_mask = mask['attention_mask']
             
              self.data_list.append(torch.tensor(id))
              self.attention_masks.append(torch.tensor(attention_mask))
             
              if mode != 'test':
                self.labels.append(row['sentiment_label'])
            # padding to the same size 
            self.data_list = pad_sequence(self.data_list, batch_first=True, padding_value=0)
            self.attention_masks = pad_sequence(self.attention_masks, batch_first=True, padding_value=0)

    def __getitem__(self, index):        
        input_ids = torch.tensor(self.data_list[index])
        attention_masks = torch.tensor(self.attention_masks[index])
        if self.mode == 'test':
            return input_ids, attention_masks

        label = torch.tensor(int(self.labels[index]))
        return input_ids, attention_masks, label

    def __len__(self):
        return len(self.data_list) 

        
dataset_train = textData('./twitter_sentiment/train.csv', mode='train')
dataset_val = textData('./twitter_sentiment/val.csv', mode='val')
dataset_test = textData('./twitter_sentiment/test.csv', mode='test')
# print(dataset_test)

"""## Loading the file """

from torch.utils.data import DataLoader

train_loader = DataLoader(dataset_train, batch_size=128, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size=128, shuffle=False)
test_loader = DataLoader(dataset_test, batch_size=128, shuffle=False)

print("There are", dataset_val.__len__(), "sentences in dataset_train.")

"""## Import Bert """

import torch
import torch.nn as nn
from transformers import BertModel

class Bert_model(nn.Module):

    def __init__(self, freeze=False):

        super(Bert_model, self).__init__()
        
        # BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # Freeze model
        
        self.flatten = nn.Sequential(
            nn.Linear(768, 100),
            nn.ReLU(),
            nn.Linear(100, 3)
        )
        if freeze:
          for param in self.bert.parameters():
              param.requires_grad = False
                             
                             
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,attention_mask=attention_mask)
                                   
        # Extract the last hidden state of the token `[CLS]` for classification task
        last = outputs[0][:, 0, :]

        out = self.flatten(last)

        return out

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Bert_model(freeze=False)
model = model.cuda()
model.to(device)

from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
import torch.optim as optim

# Total number of training steps 6 epoch
total_steps = len(train_loader) * 1
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(),lr=4e-5,eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps=total_steps)
criterion = criterion.cuda()

def train(input_data, model, criterion, optimizer):
    '''
    Argement:
    input_data -- iterable data, typr torch.utils.data.Dataloader is prefer
    model -- nn.Module, model contain forward to predict output
    criterion -- loss function, used to evaluate goodness of model
    optimizer -- optmizer function, method for weight updating
    '''
    model.train()
    loss_list = []
    total_count = 0
    acc_count = 0
    for i, data in enumerate(input_data, 0):
        if torch.cuda.is_available():
          input_ids, attention_mask, labels  = data[0].cuda(), data[1].cuda(), data[2].cuda()
 
        optimizer.zero_grad()
        out = model(input_ids,attention_mask)
        loss = criterion(out,labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
       
        _, predicted = torch.max(out.data,1)
        total_count +=labels.size(0)
        acc_count += (predicted == labels).sum().item()
        loss_list.append(loss.item())
        ########################################################################
        #                           End of your code                           #
        ########################################################################

    # Compute this epoch accuracy and loss
    acc = acc_count / total_count
    loss = sum(loss_list) / len(loss_list)
    return acc, loss

def val(input_data, model, criterion):
    model.eval()
    
    loss_list = []
    total_count = 0
    acc_count = 0
    with torch.no_grad():
        for data in input_data:
            if torch.cuda.is_available():
              input_ids, attention_mask, labels  = data[0].cuda(), data[1].cuda(), data[2].cuda()
            out = model(input_ids,attention_mask)
            loss = criterion(out, labels)
            _, predicted = torch.max(out.data,1)
            total_count +=labels.size(0)
            acc_count += (predicted == labels).sum().item()       
            loss_list.append(loss.item())
              
    acc = acc_count / total_count
    loss = sum(loss_list) / len(loss_list)
    return acc, loss

################################################################################
# You can adjust those hyper parameters to loop for max_epochs times           #
################################################################################
max_epochs = 1
log_interval = 1 # print acc and loss in per log_interval time
################################################################################
#                               End of your code                               #
################################################################################
train_acc_list = []
train_loss_list = []
val_acc_list = []
val_loss_list = []

for epoch in range(1, max_epochs + 1):
    train_acc, train_loss = train(train_loader, model, criterion, optimizer)
    val_acc, val_loss = val(val_loader, model, criterion)
    
    train_acc_list.append(train_acc)
    train_loss_list.append(train_loss)
    val_acc_list.append(val_acc)
    val_loss_list.append(val_loss)
    if epoch % log_interval == 0:
        print('=' * 20, 'Epoch', epoch, '=' * 20)
        print('Train Acc: {:.6f} Train Loss: {:.6f}'.format(train_acc, train_loss))
        print('Val Acc: {:.6f}   Val Loss: {:.6f}'.format(val_acc, val_loss))

def predict(input_data, model):
    model.eval()
    output_list = []
    with torch.no_grad():
        for data in input_data:
            input_ids, attention_mask = data[0].cuda(), data[1].cuda()
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs.data, 1)
            output_list.extend(predicted.to('cpu').numpy().tolist())
    return output_list

idx = 0

output_csv = predict(test_loader, model)
with open('result.csv', 'w', newline='') as csvFile:
    writer = csv.DictWriter(csvFile, fieldnames=['index', 'sentiment_label'])
    writer.writeheader()
    for result in output_csv:
        writer.writerow({'index':idx, 'sentiment_label':result})
        idx+=1
