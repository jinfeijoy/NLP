#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import io
import os
import re
import sys
from pandasql import sqldf
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix
import jieba
root_path = 'C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\NLP_data\\simplifyweibo_4_moods'
sys.path.append('C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\python_functions\\jl_nlp_pkg')
stop_words_path = 'C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\NLP\\recomend_system\\Chinese\\hit_stopwords.txt'

sys.path.append('C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\python_functions\\jl_nlp_pkg')
import nlpbasic.dataExploration as DataExploration


# In[10]:


weibo_data = pd.read_csv(os.path.join(root_path, "simplifyweibo_4_moods.csv"))
weibo_sample = weibo_data.sample(10000, random_state=1)
weibo_sample.to_csv(os.path.join(root_path, 'weibo_sample_4mood.csv'), index=False)


# In[49]:


import csv
import json
 
# Takes the file paths as arguments
def make_json(csvFilePath, jsonFilePath):
     
    # create a dictionary
    data = {}
     
    # Open a csv reader called DictReader
    with open(csvFilePath, encoding='utf-8') as csvf:
        csvReader = csv.DictReader(csvf)
        data = list(csvReader)
 
    # Open a json writer, and use the json.dumps()
    # function to dump data
    with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
        jsonf.write(json.dumps(data, indent=4))

make_json(os.path.join(root_path, "weibo_sample_4mood.csv"), os.path.join(root_path, "weibo_sample_4mood.json"))


# Reference: https://blog.csdn.net/qq_42103091/article/details/119978834?utm_source=app&app_version=4.16.0

# In[12]:


import pynlpir
def weibo_process(content):
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


# In[5]:


# license update for pynlpir: https://github.com/jinfeijoy/NLPIR/tree/master/License; data path: C:\Users\luoyan011\.conda\envs\nlp\Lib\site-packages\pynlpir\Data
weibo_process(weibo_data.review[0])


# In[52]:


import json
ignore_chars = ["/","@","【","】","#",":","[","]"]

def datasetProcess(org_path,save_path,stop_words):
    """
    功能：过滤出微博内容重点中文并进行分词
    """
    outcome = []
    with open(org_path,"r",encoding="utf-8") as fp:
        for idx,item in enumerate(json.load(fp)):
            print("processing item {}".format(idx))
            content = item.get("review")
            label = item.get("label")
            # content = "".join(regex.findall(chinese,content))
            seg_list = weibo_process(content)
            # seg_list = jieba.cut(content,cut_all=False)
            words = []
            for word in seg_list:
                if word in ignore_chars:
                    continue
                if word not in stop_words:
                    words.append(word)
            outcome.append({"content":words,"label":label})
    with open(save_path,"w",encoding="utf-8") as fp:
        json.dump(outcome,fp,ensure_ascii=False)


# In[53]:


print('start read stopwords data.')
stopwords = []
with open(stop_words_path, 'r', encoding='utf-8') as f:
    for line in f:
        if len(line)>0:
            stopwords.append(line.strip())
print(stopwords)


# In[54]:


datasetProcess(os.path.join(root_path, "weibo_sample_4mood.json"),os.path.join(root_path, "weibo_sample_4mood_process.json"),stopwords)


# In[5]:


f = open(os.path.join(root_path, "weibo_sample_4mood_process.json"), encoding="utf8")
sample_processed_json = json.load(f)


# In[66]:


sample_processed_json = pd.read_json(open(os.path.join(root_path, "weibo_sample_4mood_process.json"), encoding="utf8"))


# In[67]:


sample_processed_json.info


# In[9]:


import plotly.express as px
explore_label = sample_processed_json.groupby('label').agg('count').reset_index()
fig = px.pie(explore_label, values='content', names='label')
fig.show()


# In[55]:


def getWordDict(data_path, file_name, min_count=5):
    """
    功能：构建单词词典
    """
    word2id = {}
    # 统计词频
    with open(os.path.join(data_path,file_name),"r",encoding="utf-8") as fp:
        for item in json.load(fp):
            for word in item['content']:
                if word2id.get(word) == None:
                    word2id[word] = 1
                else:
                    word2id[word] += 1
    # 过滤低频词
    vocab = set()
    for word,count in word2id.items():
        if count >= min_count:
            vocab.add(word)

    # 构成单词到索引的映射词典
    word2id = {"PAD":0,"UNK":1}
    length = 2
    for word in vocab:
        word2id[word] = length
        length += 1
    with open(os.path.join(data_path, "word2id.json"),'w',encoding="utf-8") as fp:
        json.dump(word2id,fp,ensure_ascii=False)


# In[56]:


getWordDict(root_path, "weibo_sample_4mood_process.json",min_count=5)


# In[68]:


sample_processed_json.head(3)


# Word Cloud:
# * https://zhuanlan.zhihu.com/p/28954970
# * https://zhuanlan.zhihu.com/p/138356932

# In[8]:


from numpy import array, concatenate, atleast_1d
all_tokens = concatenate([atleast_1d(a) for a in sample_processed_json.content.apply(lambda x: ' '.join(x))])
all_tokens = " ".join(all_tokens)


# In[1]:


def generate_chinese_word_cloud(text, img_link, font_path, stopwords = '',
                                color_control = False, contour_color = 'white', save = False):
    """

    :param text: corpus with space as split
    :param img_link: background image path
    :param font_path: chinese font path (e.g. simsun.ttf)
    :param stopwords: stopwords list
    :param color_control: True: use same color as background, false: use default color
    :param contour_color: color name of contour, default is white, can be changed to e.g. 'green','pink',etc
    :param save: to save img to current folder or not, default is false
    :return:
    """
    backgroud = np.array(Image.open(img_link))

    wc = WordCloud(width=800, height=800,
            background_color='white',
            mode='RGB',
            mask=backgroud, #添加蒙版，生成指定形状的词云，并且词云图的颜色可从蒙版里提取
            contour_width=3,
            contour_color=contour_color,
            max_words=500,
            stopwords=STOPWORDS.add(stopwords),#内置的屏蔽词,并添加自己设置的词语
            font_path=font_path,
            max_font_size=150,
            relative_scaling=0.6, #设置字体大小与词频的关联程度为0.4
            random_state=50,
            scale=2
            ).generate(text)
    if color_control == True:
        image_color = ImageColorGenerator(backgroud)#设置生成词云的颜色，如去掉这两行则字体为默认颜色
        wc.recolor(color_func=image_color)

    plt.imshow(wc) #显示词云
    plt.axis('off') #关闭x,y轴
    plt.show()#显示
    if save == True:
        wc.to_file('word_cloud.jpg')


# In[9]:


img_link = os.path.join(root_path, 'nezha.jpg')
font_path = "C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\NLP\\recomend_system\\Chinese\\simsun.ttf"
DataExploration.generate_chinese_word_cloud(all_tokens, img_link, font_path ,stopwords = '', color_control = True)


# ## Train Model
# https://www.pythonf.cn/read/59661
# 
# https://machinelearningmastery.com/develop-word-embedding-model-predicting-movie-review-sentiment/
# 
# https://www.cxyzjd.com/article/qq_35386727/96343253
# 
# https://medium.com/@black_swan/%E7%94%A8%E7%B6%AD%E5%9F%BA%E8%AA%9E%E6%96%99%E8%A8%93%E7%B7%B4-word2vec-%E5%92%8C-fasttext-embedding-25ede5b15994
# 
# https://wshuyi.medium.com/%E5%A6%82%E4%BD%95%E7%94%A8-python-%E5%92%8C-gensim-%E8%B0%83%E7%94%A8%E4%B8%AD%E6%96%87%E8%AF%8D%E5%B5%8C%E5%85%A5%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B-87b6183229b7

# In[13]:


moods = {0: '喜悦', 1: '愤怒', 2: '厌恶', 3: '低落'}

print('微博数目（总体）：%d' % weibo_sample.shape[0])

for label, mood in moods.items(): 
    print('微博数目（{}）：{}'.format(mood,  weibo_sample[weibo_sample.label==label].shape[0]))


# In[69]:


weibo_sample.head(3)


# In[71]:


def preprocess_text(content_lines, sentences, category):
    for line in content_lines:
        try:
            segs=jieba.lcut(line)
            segs = filter(lambda x:len(x)>1, segs)
            segs = [v for v in segs if not str(v).isdigit()]#去数字
            segs = list(filter(lambda x:x.strip(), segs)) #去左右空格
            segs = filter(lambda x:x not in stopwords, segs)
            temp = " ".join(segs)
            if(len(temp)>1):
                sentences.append((temp, category))
        except Exception:
            print(line)
            continue


# In[154]:


import random
data_label_0_content = weibo_sample[weibo_sample.label==0]['review'].values.tolist()
data_label_1_content = weibo_sample[weibo_sample.label==1]['review'].values.tolist()
data_label_2_content = weibo_sample[weibo_sample.label==2]['review'].values.tolist()
data_label_3_content = weibo_sample[weibo_sample.label==3]['review'].values.tolist()
sentences = []
preprocess_text(data_label_0_content, sentences, 0)
preprocess_text(data_label_1_content, sentences, 1)
preprocess_text(data_label_2_content, sentences, 2)
preprocess_text(data_label_3_content, sentences, 3)
random.shuffle(sentences)


# In[106]:


def getWordDict(document, data_path, min_count=5):
    """
    功能：构建单词词典
    """
    word2id = {}
    # 统计词频
    for word in sum([a_tuple[0].split(' ') for a_tuple in document],[]):
        if word2id.get(word) == None:
            word2id[word] = 1
        else:
            word2id[word] += 1
            
    # 过滤低频词
    vocab = set()
    for word,count in word2id.items():
        if count >= min_count:
            vocab.add(word)

    # 构成单词到索引的映射词典
    word2id = {"PAD":0,"UNK":1}
    length = 2
    for word in vocab:
        word2id[word] = length
        length += 1
    with open(os.path.join(data_path, "word2id.json"),'w',encoding="utf-8") as fp:
        json.dump(word2id,fp,ensure_ascii=False)


# In[133]:


getWordDict(sentences,root_path, min_count=1)


# In[99]:


sum([a_tuple[0].split(' ') for a_tuple in sentences[0:5]],[])


# In[243]:


from sklearn.model_selection import train_test_split
x, y = zip(*sentences)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=1234)#所以把原数据集分成训练集的测试集，咱们用sklearn自带的分割函数。
from sklearn.model_selection import train_test_split
x, y = zip(*sentences)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=1234)
 


# In[186]:


print(x_train[0:3])
print(y[0:3])


# ### Vectorization + Keras (Performance Not Good)

# In[156]:


#抽取特征，我们对文本抽取词袋模型特征
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer(
    analyzer='word', #tokenise by character ngrams
    max_features=4000,  #keep the most common 1000 ngrams
)
vec.fit(x_train)


# In[157]:


# 设置参数
max_features = 5001
maxlen = 100
batch_size = 32
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 10
nclasses = 4


# In[158]:


from keras.utils import np_utils
from keras.preprocessing import sequence
x_train = vec.transform(x_train)
x_test = vec.transform(x_test)
x_train = x_train.toarray()
x_test = x_test.toarray()
y_train = np_utils.to_categorical(y_train,nclasses)
y_test = np_utils.to_categorical(y_test,nclasses)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)


# In[159]:


import keras
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()


# In[160]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
history = LossHistory()
print('Build model...')
model = Sequential()

model.add(Embedding(max_features,
                        embedding_dims,
                        input_length=maxlen))
model.add(Dropout(0.5))
model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
model.add(GlobalMaxPooling1D())
model.add(Dense(hidden_dims))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(nclasses))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),callbacks=[history])


# In[161]:


x_test[1]


# In[162]:


import matplotlib.pyplot as plt
LossHistory.loss_plot(history,'epoch')


# In[163]:


model.predict(x_test)


# This approach is not good, we need to try other approaches.
# ### Word Embedding & gensim

# In[20]:


wiki_embedding_path = 'C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\NLP_data\\zh_wiki_word2vec_300\\zh_wiki_word2vec_300.txt'


# In[21]:


from numpy import array,asarray,zeros
# load the whole embedding into memory
embeddings_index = dict()
# download glove word embedding first and then load it with the following code
f = open(wiki_embedding_path, encoding = 'utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = asarray(values[1:], dtype = 'float32')
    embeddings_index[word] = coefs
f.close
print('loaded %s word vectors.' % len(embeddings_index))


# In[135]:


f = open(os.path.join(root_path, "word2id.json"), encoding="utf8")
data_vocabulary = json.load(f)


# In[136]:


data_vocabulary


# In[180]:


# Initialize a matrix with zeros having dimensions equivalent to vocab size and 100
import json
vocab_size = len(data_vocabulary) + 1

embedding_matrix = zeros((vocab_size, 300))
vocab_list = [None] * vocab_size

with open(os.path.join(root_path, 'word2id.json'),"r",encoding="utf-8") as fp:
        for idx_word,word in enumerate(json.load(fp)):
            word_vector = embeddings_index.get(word)
            if word_vector is not None:
                embedding_matrix[idx_word] = word_vector
                vocab_list[idx_word] = word

# for word, idx_word in t.word_index.items():
#     word_vector = embeddings_index.get(word)
#     if word_vector is not None:
#         embedding_matrix[idx_word] = word_vector
print('word:', vocab_list[3])
print('Embedding:\n', embedding_matrix[3])
print('length of embedding matrix is:', len(embedding_matrix))
print('vocabulary size is %s.' % vocab_size)


# In[111]:


from scipy.spatial import distance
def find_closest_embeddings(embedding):
    return sorted(embeddings_index.keys(), key = lambda word: distance.euclidean(embeddings_index[word], embedding))


# In[62]:


distance.euclidean(embeddings_index['造成'], embedding_matrix[data_vocabulary.get('造成')])


# In[112]:


find_closest_embeddings(embedding_matrix[data_vocabulary.get('造成')])


# In[137]:


x_train[3].split(' ')


# In[130]:


explore_size = [len(a_tuple[0].split(' ')) for a_tuple in sentences]
from matplotlib import pyplot as plt
plt.hist(explore_size,  alpha=0.5)


# In[168]:


from keras.preprocessing.sequence import pad_sequences
max_length = 25
encoded_docs = [list(map(data_vocabulary.get, i.split(' '))) for i in x_train]
print('Encoding:\n', encoded_docs[0])
print('\nText:\n', list(x_train)[0])
print('\nWord Indices:\n', [(data_vocabulary.get(i), i) for i in x_train[0].split(' ')])

padded_docs = pad_sequences(encoded_docs, maxlen = max_length, padding = 'post')
# for sequences shorter than maxlen add 0 to the end of the sequence (when pdding = 'post'), for sequence longer than maxlen, trunc senquence from the end of the sequence


# In[167]:


encoded_test_doc = [list(map(data_vocabulary.get, i.split(' '))) for i in x_test]
padded_test_docs = pad_sequences(encoded_test_doc, maxlen = max_length, padding = 'post')
print("===== test set =====")
print(padded_test_docs)


# In[254]:


from keras.models import Sequential
from keras.layers import Dense,Flatten,Embedding
embedding_dim = 300
model = Sequential(
    [
        Embedding(vocab_size, embedding_dim, weights = [embedding_matrix], input_length = max_length, trainable = False),
        Flatten(),
        Dense(embedding_dim, activation="relu", name="layer1"),
        Dense(4, activation = 'softmax', name="layer2")
        
    ]
)
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])
print(model.summary())


# In[227]:


print(padded_docs)
print(y_train)
print(padded_test_docs)
print(y_test)


# In[253]:


print(y_train_t)


# In[255]:


from tensorflow.keras.utils import to_categorical
padded_docs = np.array(padded_docs)
padded_test_docs = np.array(padded_test_docs)
# y_train = np.array(y_train)
# y_test = np.array(y_test)
y_train_t = pd.get_dummies(y_train).values
y_test_t = pd.get_dummies(y_test).values

history = model.fit(padded_docs, 
                    y_train_t, 
                    epochs = 10, 
                    verbose = 1, 
                    batch_size = 32, 
                    validation_data = (padded_test_docs, y_test_t))


# In[256]:


acc = history.history['acc']
print ("Accuracy history: ",acc)
val_acc = history.history['val_acc']
print("\nValidation history: ",val_acc)
loss = history.history['loss']
val_loss = history.history['val_loss']


# In[257]:


# plot loss rate and accuracy
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, color = 'r', label = 'Training acc')
plt.plot(epochs, val_acc, color = 'b', label = 'Validation acc')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.figure()
plt.plot(epochs, loss, color = 'r', label = 'Training loss')
plt.plot(epochs, val_loss, color = 'b', label = 'Validation loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()


# In[259]:


label_list = [0,1,2,3]
pred_test = model.predict(padded_test_docs)
pred_test


# In[260]:


print(len([label_list[np.argmax(i)] for i in y_test]))
print(len([label_list[np.argmax(i)] for i in pred_test]))


# In[261]:


from sklearn.metrics import accuracy_score
accuracy_score([label_list[np.argmax(i)] for i in y_test], [label_list[np.argmax(i)] for i in pred_test])


# In[263]:


from sklearn.metrics import confusion_matrix, classification_report
print(classification_report([label_list[np.argmax(i)] for i in y_test],[label_list[np.argmax(i)] for i in pred_test]))


# In[283]:


def predict_for_lime(text_array):
    encoded =[list(map(data_vocabulary.get, i.split(' '))) for i in text_array]
    text_data = pad_sequences(encoded, maxlen=max_length,padding='post')
    pred=model.predict(text_data)
    return pred

# test the predicition function
print ("Verify if predictions are correct for the function")
print(predict_for_lime([x_test[0],x_test[533]]))
print(y_test[0], y_test[533])

