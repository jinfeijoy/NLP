#!/usr/bin/env python
# coding: utf-8

# In[31]:


import argparse
import thulac
import json
import os

from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
stop_words_path = 'C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\NLP\\recomend_system\\Chinese\\hit_stopwords.txt'

datapath = 'C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\GPT2-Chinese\\data'
vocab_size = 5000


# ## Process Text and Generate Vocabulary

# In[36]:


import re
import pynlpir
def text_process(content):
    """
    功能：清洗微博内容并分词
    """
    processed_content = []
    # Replaces URLs with the word [URL]
    content = re.sub(r'(https?|ftp|file|www\.)[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', '[URL]', content)
    # Replaces Email with the word [URL]
    content = re.sub(r'[a-zA-Z0-9_-]+@[a-zA-Z0-9_-]+[\.][a-zA-Z0-9_-]+', '[URL]', content)
    # Replaces user with the word FORWARD
    content = re.sub(r'(\/\/){0,1}@.*?(：|:| )', '[FORWARD]', content)
    # Replaces number  with the word [N]
    content = re.sub(r'\d+', '[N]', content)
    # Replace 2+ dots with space
    content = re.sub(r'[\.。…]{2,}', '。', content)
    # Replace 2+ ~~ 为 ~
    content = re.sub(r'~{2,}', '~', content)
    # Replace 2+ 叹号 为 一个叹号
    content = re.sub(r'[!！]{2,}', '!', content)
    # Replace 2+ 叹号 为 一个问号
    content = re.sub(r'[？?]{2,}', '?', content)
    # 去掉 //
    content = re.sub(r'//', ' ', content)
    # 去掉 引号
    content = re.sub(r'["“”\'‘’]', '', content)

    pynlpir.open(encoding='utf_8', encoding_errors='ignore')
    segments = pynlpir.segment(content, pos_tagging=False)
    i = 1
    count = len(segments) - 1
    for segment in segments:
        if re.match(r'\s+', segment):  # 过滤掉空格
            i = i + 1
            continue
        segment = re.sub(r'@[\S]+', '[USER_MENTION]', segment)
        processed_content.append(segment.strip())
        if (i == count) & (segment == '[USER_MENTION]'):  # 过滤掉最后一个单独的字
            break
        i = i + 1
    pynlpir.close()
    return processed_content


def datasetProcess(org_path,save_path,stop_words):
    """
    功能：过滤出微博内容重点中文并进行分词
    """
    outcome = []
    with open(org_path,"r",encoding="utf-8") as fp:
        for idx,item in enumerate(json.load(fp)):
            print("processing item {}".format(idx))
            content = item
            # content = "".join(regex.findall(chinese,content))
            seg_list = weibo_process(content)
            # seg_list = jieba.cut(content,cut_all=False)
            words = []
            for word in seg_list:
                if word in ignore_chars:
                    continue
                if word not in stop_words:
                    words.append(word)
            outcome.append(words)
    with open(save_path,"w",encoding="utf-8") as fp:
        json.dump(outcome,fp,ensure_ascii=False)

print('start read stopwords data.')
stopwords = []
with open(stop_words_path, 'r', encoding='utf-8') as f:
    for line in f:
        if len(line)>0:
            stopwords.append(line.strip())
print(stopwords)


# In[37]:


ignore_chars = ["/","@","【","】","#",":","[","]"]
datasetProcess(os.path.join(datapath, "train.json"),
               os.path.join(datapath, "train_process.json"),
               stopwords)


# In[38]:


f = open(os.path.join(datapath, 'train.json'), 'r', encoding = 'utf-8')
lines = json.load(f)
lines[0]


# In[39]:


f = open(os.path.join(datapath, 'train_process.json'), 'r', encoding = 'utf-8')
lines = json.load(f)
lines[0]


# In[23]:





# In[41]:


def main():

    tokenizer = Tokenizer(num_words=vocab_size)

    f = open(os.path.join(datapath, 'train_process.json'), 'r', encoding = 'utf-8')
    lines = json.load(f)

    tokenizer.fit_on_texts(lines)
    vocab = list(tokenizer.index_word.values())
    pre = ['[SEP]', '[CLS]', '[MASK]', '[PAD]', '[UNK]']
    vocab = pre + vocab
    with open(os.path.join(datapath, 'vocab.txt'), 'w', encoding = 'utf-8') as f:
        for word in vocab[:vocab_size + 5]:
            f.write(word + '\n')


if __name__ == "__main__":
    main()

