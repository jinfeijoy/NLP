#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import json, pprint, time, os, io, re, sys
from urllib import parse
import requests
from bs4 import BeautifulSoup
from pandasql import sqldf
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix
import jieba
datapath = 'C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\GPT2-Chinese\\data'
sys.path.append('C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\python_functions\\jl_nlp_pkg')

import nlpbasic.dataExploration as DataExploration
import time


# In[59]:


url = 'https://www.qianxuntxt.com/xiaoshuo11491/'
response = requests.get(url, headers = headers)
soup = BeautifulSoup(response.text, 'html.parser')
a_elms = soup.find_all('a', href=True)
a_elms
for a in a_elms:
    if a.get('href')[0].isdigit():
        print(a.get('href'))


# In[61]:


listData = []
url = 'https://www.miaojiang8.net/7_7336'
sub_url = 'https://www.miaojiang8.net'

headers = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36'
}

def getMainLinks():
    response = requests.get(url, headers = headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    a_elms = soup.find_all('a', href=True)
    for a in a_elms:
        if a.get('href').endswith('html'):
            if a.get('href').startswith('/'):
                listData.append({
                    'title': a.text.encode('iso-8859-1').decode('gbk'),
                    'link': sub_url + parse.unquote(a.get('href'))
                })
            if a.get('href').startswith('http'):
                listData.append({
                    'title': a.text.encode('iso-8859-1').decode('gbk'),
                    'link': parse.unquote(a.get('href'))
                })
            if a.get('href')[0].isdigit():
                listData.append({
                    'title': a.text.encode('iso-8859-1').decode('gbk'),
                    'link': url + '/' + parse.unquote(a.get('href'))
                })
            
    pprint.pprint(listData)
    
                             
def saveJson():
    fp = open(os.path.join(datapath,'cuizhuang.json'),'w',encoding = 'utf-8')
    fp.write(json.dumps(listData, ensure_ascii=False))
    fp.close()
    
def writeTxt():
    listContent = []
    fp = open(os.path.join(datapath, 'cuizhuang.json'),'r',encoding = 'utf-8')
    strJson = fp.read()
    fp.close()
    folderPath = os.path.join(datapath,'cuizhuang_txt')
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
    listResult = json.loads(strJson)
    for i in range(len(listResult)):
        response = requests.get(listResult[i]['link'], headers = headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        print(i)
        print(listResult[i]['title'])
        div = soup.select_one('div#content')
        strContent = div.text.encode('iso-8859-1').decode('gbk')
        strContent = strContent.replace(" ", "")
        strContent = strContent.replace("\r", "")
#         strContent = strContent.replace("\n", "")   
        
        fileName = f"{listResult[i]['title']}.txt"
        fp = open(f"{folderPath}/{fileName}","w",encoding="utf-8")
        fp.write(strContent)
        fp.close()
        
        listContent.append(strContent)
        time.sleep(2)
        
    fp = open(os.path.join(datapath, 'train.json'), 'w', encoding = 'utf-8')
    fp.write(json.dumps(listContent, ensure_ascii = False))
    fp.close
                    
if __name__ == '__main__':
    getMainLinks()
    saveJson()
    writeTxt()


# In[3]:


def listToString(s): 
    
    # initialize an empty string
    str1 = " " 
    
    # return string  
    return (str1.join(s))
txt_path = os.path.join(datapath,'cuizhuang_txt_old')

os.chdir(txt_path)
all_files = sorted(filter(os.path.isfile, os.listdir('.')), key=os.path.getmtime)

listContent = []
for file in all_files:
    f = open(os.path.join(txt_path,file), encoding = 'utf-8')
    chapter_txt = f.readlines()
    chapter_txt = listToString(chapter_txt).replace('牋牋','').replace('\n','')
    listContent.append(chapter_txt)

fp = open(os.path.join(datapath, 'train.json'), 'w', encoding = 'utf-8')
fp.write(json.dumps(listContent, ensure_ascii = False))
fp.close


# In[6]:


f = open(os.path.join(txt_path,'第十一章 郑珍语.txt'), encoding = 'utf-8')
chapter_txt = f.readlines()
chapter_txt = listToString(chapter_txt).replace('牋牋','')
chapter_txt


# In[11]:


def listToString(s): 
    
    # initialize an empty string
    str1 = " " 
    
    # return string  
    return (str1.join(s))

txt_path = os.path.join(datapath,'cuizhuang_txt_old')
os.chdir(txt_path)
all_files = sorted(filter(os.path.isfile, os.listdir('.')), key=os.path.getmtime)[718:775]
all_files

listContent = []
for file in all_files:
    f = open(os.path.join(txt_path,file), encoding = 'utf-8')
    chapter_txt = f.readlines()
    print(file)
    chapter_txt = listToString(chapter_txt).replace('牋牋','')
    listContent.append(chapter_txt)
    
fp = open(os.path.join(datapath, 'cuizhuang_read.txt'), 'w', encoding = 'utf-8')
listContent = listToString(listContent)
fp.write(listContent)
fp.close


# In[45]:


response = requests.get('https://www.miaojiang8.net/7_7336/7313924.html', headers = headers)
soup = BeautifulSoup(response.text, 'html.parser')
div = soup.select_one('div#content')
div.encode('iso-8859-1').decode('gbk')

