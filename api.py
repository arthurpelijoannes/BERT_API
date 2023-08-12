#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 18:18:30 2023

@author: arthur
"""
import flask
from flask import Flask,render_template,request
import os 
import re
from bs4 import BeautifulSoup
import torch
from transformers import BertModel, BertTokenizer
import pandas as pd

def get_indexes (boolean_tensor) :
    indexes = []
    for i in range (len(boolean_tensor[0])) :
        if boolean_tensor[0, i] :
            indexes.append(i)
    return indexes
            
path = ''    
cons = 'bcdfghjklmnpqrstvwxz'
voy = 'aeiouy'

category_list = ['.net', 'algorithm', 'android', 'angular', 'angularjs', 'arrays',
       'asp.net', 'asp.net-mvc', 'bash', 'c', 'c#', 'c++', 'c++11', 'css',
       'django', 'gcc', 'git', 'google-chrome', 'haskell', 'html', 'ios',
       'iphone', 'java', 'javascript', 'jquery', 'json',
       'language-lawyer', 'linux', 'macos', 'multithreading', 'mysql',
       'node.js', 'objective-c', 'performance', 'php', 'python',
       'python-3.x', 'r', 'reactjs', 'regex', 'ruby', 'ruby-on-rails',
       'spring', 'sql', 'sql-server', 'string', 'swift', 'visual-studio',
       'windows', 'xcode']


# df = pd.read_csv('Documents/Formation/Projet 5/cleaned_data_v0.csv')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class BERTClass(torch.nn.Module):

    def __init__(self):

        super(BERTClass, self).__init__()

        self.l1 = BertModel.from_pretrained('bert-base-uncased')

        self.l2 = torch.nn.Dropout(0.3)

        self.l3 = torch.nn.Linear(768, 50)

    def forward(self, ids, mask, token_type_ids):

        output_1 = self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids).pooler_output

        output_2 = self.l2(output_1)

        output = self.l3(output_2)

        return output

model = torch.load(path + 'BERT_model.pt', map_location = device)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')



# body = str(df['Body_cleaned'][1])
# model_input = re.sub(r'\d', '', re.sub('\W+',' ', BeautifulSoup(body, 'html.parser').get_text())).replace('   ', ' ').replace('  ', ' ')
# max_len = 256
# body = " ".join(body.split())



# inputs = tokenizer.encode_plus(
#     body,
#     None,
#     add_special_tokens=True,
#     max_length=256,
#     padding='max_length',
#     return_token_type_ids=True,
#     truncation=True
# )

# ids = torch.tensor(inputs['input_ids'], dtype=torch.long)[None, :]
# mask = torch.tensor(inputs['attention_mask'], dtype=torch.long)[None, :]
# token_type_ids = torch.tensor(inputs["token_type_ids"], dtype=torch.long)[None, :]

# outputs = model(ids, mask, token_type_ids)
# cat = category_list[torch.argmax(outputs)]

# print('the post belongs to :')
# for index in get_indexes(torch.nn.Sigmoid()(outputs)>0.5) :
#     print(category_list[index])
    
    
    
    
app = Flask(__name__, template_folder='templates', static_folder='statics')
app.debug = True
 
@app.route('/')
def form():
    return render_template('form.html')
 
@app.route('/data/', methods = ['POST', 'GET'])
def data():
    if request.method == 'GET':
        return f"The URL /data is accessed directly. Try going to '/form' to submit form"
    if request.method == 'POST':
        post_text = request.form.get("multiliner")
        print(os.listdir())
        body = post_text
        model_input = re.sub(r'\d', '', re.sub('\W+',' ', BeautifulSoup(body, 'html.parser').get_text())).replace('   ', ' ').replace('  ', ' ')
        max_len = 256
        body = " ".join(body.split())
    
    
    
        inputs = tokenizer.encode_plus(
            body,
            None,
            add_special_tokens=True,
            max_length=256,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
    
        ids = torch.tensor(inputs['input_ids'], dtype=torch.long)[None, :]
        mask = torch.tensor(inputs['attention_mask'], dtype=torch.long)[None, :]
        token_type_ids = torch.tensor(inputs["token_type_ids"], dtype=torch.long)[None, :]
    
        outputs = model(ids, mask, token_type_ids)
        cat = category_list[torch.argmax(outputs)]
        
        form_data = 'the post belongs to :\n'
        for index in get_indexes(torch.nn.Sigmoid()(outputs)>0.5) :
            form_data += category_list[index] +'\n'
        
        
        return form_data
 
 
app.run(host='localhost', port=5000)
