## [Modern Approach in NLP Processing](https://slds-lmu.github.io/seminar_nlp_ss20/)
## Chaper2 Introduction: DL for NLP
  - Word Embedding and Nueral Netword Model
    - Word Embedding
      - Word embeddings use dense vector representations for words. That means they map each word to a continuous vector with n dimensions. The distance in the vector space denotes semantic (dis)similarity.
      - Practically all NLP projects these days build upon word embeddings, since they have a lot of advantages compared to the aforementioned representations
      - Word2vec algorithm (Continous Bag-Of-Words (CBOW) and Skip-gram) and the GloVe algorithm for calculating word embeddings became more and more popular,applying this technique as a model input has become one of the most common text processing methods.
    - RNN: Recurrent neural networks or RNNs are a special family of neural networks which were explicitely developed for modeling sequential data like text. RNNs process a sequence of words or letters by going through its elements one by one and capturing information based on the previous elements.One particular reason why recurrent networks have become such a powerful technique in processing sequential data is parameter sharing.
    ![image](https://user-images.githubusercontent.com/16402963/119266494-0b6d7a00-bbb9-11eb-81a5-f36791fd714a.png)
    - CNN: convolutional neural networks (CNN) are widely used in computer vision, utilizing CNN to word embedding matrices and automatically extract features to handle NLP tasks appeared inevitable
    ![image](https://user-images.githubusercontent.com/16402963/119266509-188a6900-bbb9-11eb-84be-6eb3d28e9152.png)
    
## Chaper3 Foundations/Applications of Modern NLP
  - The Evolution of Word Embeddings
    - Bag-of-Words is especially useful if the number of distinct words is small and the sequence of the words doesn’t play a key role, like in sentiment analysis. The major drawback of these methods is that there is no notion of similarity between words.
  - Methods to Obtain Word Embeddings
    - The basic idea behind learning word embeddings is the so called distributional hypothesis. It states that words that occur in the same contexts tend to have similar meanings. 
    - word2vec algorithms outperform a lot of other standard NNLM models
    - Skip-gram works well with small amounts of training data and has good representations for words that are considered rare, whereas CBOW trains several times faster and has slightly better accuracy for frequent words.
    - [GloVe](https://nlp.stanford.edu/projects/glove/) stands for Global Vector word representation, which emphasizes the global character of this model. The model builds on the possibility to derive semantic relationships between words from the co-occurrence matrix and that the ratio of co-occurrence probabilities of two words with a third word is more indicative of semantic association than a direct co-occurrence probability (see Pennington et al. (2014)). 
  - Hyperparameter Tuning and System Design Choices
    - in a lot of cases, changing the setting of a single hyperparameter could yield a greater increase in performance than switching to a better algorithm or training on a larger corpus
    - These are parameters like the number of epochs, batch-size, learning rate, embedding size, window size, corpus size et cetera.
      - embedding size: In practice word embedding vectors with dimensions around 50 to 300 are usually used as a rule of thumb (see Goldberg (2016)). Pennington et al. (2014) compare performance of the GloVe model for embedding sizes from 1 to 600 for different evaluation tasks (semantic, syntactic and overall). They found that after around 200 dimensions the performance increase begins to stagnate.
      - Context Window: A window size of 5 is commonly used to capture broad topic/domain information like what other words are used in related discussions (i.e. dog, bark and leash will be grouped together, as well as walked, run and walking), whereas smaller windows contain more specific information about the focus word and produce more functional and syntactic similarities
      - Document Context: One can either consider this as using very large window sizes or, as in the doc2vec algorithm 
      - Subsampling of Frequent Words: Very frequent words are often so-called stop-words, like the or a, which do not provide much information.
      - Negative Sampling:  based on the skip-gram algorithm, but it optimizes a different objective. It maximizes a function of the product of word and context pairs that occur in the training data, and minimizes it for negative examples of word and context pairs that do not occur in the training corpus.
      - Subword Information: [fastText](https://fasttext.cc/docs/en/english-vectors.html) (was introduced in 2016) performs well when having data with a large number of rare words. for example, take the word, planning with n=3, the fastText representation of this word is <pl, pla, lan, ann, nni, nin, ing, ng>, where the angular brackets indicate the beginning and end of the word.
      - Phrase representation: there are many words which will only be meaningful in combination with other words, or which change meaning completely when paired up with another word. 
  - Outlook and Resources
    - When calculating word embeddings, the word order is not taken into account. For some NLP tasks like sentiment analysis, this does not pose a problem. But for other tasks like translation, word order can not be ignored. Recurrent neural networks, which will be presented in the following chapter, are one of the tools to face this difficulty
    - a lot of words with two or more different meanings.ELMO, uses contextualized embeddings to solve this problem.
    - First of the word embeddings are mostly learned from text corpora from the internet, therefore they learn a lot of stereotypes that reflect everyday human culture.
    - there are some domains and languages for which only little training data exists on the internet

## Chaper4 RNN
- Structure and Training of Simple RNNs: RNNs are powerful models for sequential data. Recurrent neural networks (RNNs) enable to relax the condition of non-cyclical connections in the classical feedforward neural networks. This means, while simple multilayer perceptrons can only map from input to output vectors, RNNs allow the entire history of previous inputs to influence the network output. 
- Gated RNNs: Main feature of gated RNNs is the ability to store long-term memory for a long time and at the same time to account for new inputs as effectively as possible. In modern NLP, two types of gated RNNs are used widely: Long Short-Term Memory networks and Gated Recurrent Units.
  - LSTM: Instead of a simple hidden unit that combines inputs and previous hidden states linearly and outputs their nonlinear transformation to the next step, hidden units are now extended by special input, forget and output gates that help to control the flow of information.
  ![image](https://user-images.githubusercontent.com/16402963/119271960-dd486400-bbd1-11eb-8919-02d3ab842afc.png)
  - Gated Recurrent Units (GRU) whose structure is simpler than that of LSTM because they have only two gates, namely reset and update gate.
  ![image](https://user-images.githubusercontent.com/16402963/119274374-99f3f280-bbdd-11eb-9d67-352e51191d55.png)
- Extensions of Simple RNNs:
  - Deep RNN: Deep RNNs with several hidden layers may improve model performance significantly. 
    - Stacked RNNs
    - Deep Transition RNNs
  - Encoder-Decoder Architecture: The problem of mapping variable-length input sequences to variable-length output sequences is known as Sequence-to-Sequence or seq2seq learning in NLP. Although originally applied in machine translation tasks (Sutskever, Vinyals, and Le (2014), Cho et al. (2014)), the seq2seq approach achieved state-of-the-art results also in speech recognition (Prabhavalkar et al. 2017) and video captioning (Venugopalan et al. 2015)
  ![image](https://user-images.githubusercontent.com/16402963/119274579-93b24600-bbde-11eb-9e63-bd75b6e41ee3.png)
  
## Chaper5 CNN
- Basic Architecture of CNN: a convolutional neural network includes successively an input layer, multiple hidden layers, and an output layer, the input layer will be dissimilar according to various applications. The hidden layers, which are the core block of a CNN architecture, consist of a series of convolutional layers, pooling layers, and finally export the output through the fully-connected layer.
  ![image](https://user-images.githubusercontent.com/16402963/119274764-8c3f6c80-bbdf-11eb-8d7b-8393608f787b.png)
  - Convolutional Layer: The convolutional layer is the core building block of a CNN. In short, the input with a specific shape will be abstracted to a feature map after passing the convolutional layer. a set of learnable filters (or kernels) plays an important role throughout this process. 
    - Convolutional Layer
    ![image](https://user-images.githubusercontent.com/16402963/119274887-29020a00-bbe0-11eb-9e33-89bab4f4c4f8.png)
    - Dilated convolution
    ![image](https://user-images.githubusercontent.com/16402963/119274910-43d47e80-bbe0-11eb-9777-a40de6e08f9e.png)
  - ReLU layer: ReLU is the most commonly used activation function in neural networks, especially in CNNs. (Krizhevsky, Sutskever, and Hinton 2012) because of its two properties: Non-linearity; Non-Saturation.
    - A non-linear layer (or activation layer) will be the subsequent process after each convolutional layer and the purpose of which is to introduce non-linearity to the neural networks because the operations during the convolutional layer are still linear (element-wise multiplications and summations). 
  - Pooling layer: The purpose of the pooling layer is to reduce progressively the spatial size of the feature map, which is generated from the previous convolutional layer, and identify important features.
    - max pooling is the most commonly used function(Scherer, Müller, and Behnke (2010)), critical reason to add max pooling to cnn is: Reducing computation complexity; Controlling overfitting.
    - Average pooling is usually used for topic models. If a sentence has different topics and the researchers assume that max pooling extracts insufficient information, average pooling can be considered as an alternative.
    - Dynamic pooling has an ability to dynamically adjust the number of features according to the network structure. words far away in the sentence also have interactive behavior (or some kind of semantic connection). Eventually, the most important semantic information in the sentence is extracted through the pooling layer.
  - Fully-connected layer: In order to improve the CNN network performance, the excitation function of each neuron in the fully connected layer generally uses the ReLU function.
- CNN for sentence classification
  - CNN-rand/CNN-static/CNN-non-static/CNN-multichannel
    ![image](https://user-images.githubusercontent.com/16402963/119275274-0d97fe80-bbe2-11eb-9c8b-c6bb0b1c2663.png) 
    - CNN-rand: All words are randomly initialized and then modified during training.
    - CNN-static: A model with pre-trained word vectors by using word2vec and keep them static.
    - CNN-non-static: A model with pre-trained word vectors by using word2vec and these word vectors are fine-tuned for each task.
    - CNN-Multichannel: A model with two channels generated by two sets of words vectors and each filter is employed to both channels.
  - Character-level ConvNets: The model architecture with 6 convolutional layers and 3 fully-connected layers (9 layers deep) is relatively more complex. Different from the previous word-based Convolutional neural networks (ConvNets), this model is at character-level by using character quantization.
    ![image](https://user-images.githubusercontent.com/16402963/119275376-9f077080-bbe2-11eb-80ce-4a210be49581.png)
  - Very Deep CNN
    - a CNN model with deep architectures of many convolutional layers is developed, the significant difference of which from the previous model architecture is that this model applies much deeper architectures (i.e. using up to 29 convolutional layers), in order to learn hierarchical representations of whole sentences.
    ![image](https://user-images.githubusercontent.com/16402963/119275535-a713e000-bbe3-11eb-9481-3cf72c5f04ff.png)
    ![image](https://user-images.githubusercontent.com/16402963/119275554-c9a5f900-bbe3-11eb-91e4-949743046dc3.png)
   - Deep Pyramid CNN: difference from the previous deep CNN is that this model is constructed of a low-complexity word-level deep CNN architecture. 
    ![image](https://user-images.githubusercontent.com/16402963/119275638-38835200-bbe4-11eb-94af-02fbe94decd6.png)
- Experimental Evaluation
  - DPCNN achieves an outstanding performance, which shows the lowest error rate in six of the eight datasets.
  - Shallow CNN, which possesses the simplest model architecture with only one convolutional layer, performs even better than other deep models.
  - By mere comparison of character-level CNN (Zhang, Zhao, and LeCun (2015)) and word-level CNN (Zhang, Zhao, and LeCun (2015)), the result will be that words-level CNN performs better in the smaller datasets and character-level CNN performs better in the bigger datasets.
- Conclusion:
  - word-level CNN possesses higher accuracy presented in lower error rate in comparison of character-level CNNs in general
  - it is essential to improve the accuracy of complex and deep models by introducing appropriate components (e.g. shortcut connections).

## Chapter 6 Introduction: Transfer Learning for NLP
- What is transfer learning: Transfer learning allows us to deal with the learning of a task by using the existing labeled data of some related tasks or domains
  ![image](https://user-images.githubusercontent.com/16402963/119275906-cdd31600-bbe5-11eb-9c39-ee3d96c5a2fd.png)
- (Self-)attention
  - A Transformer still consists of the typical encoder-decoder setup but uses a novel new architecture for both. The encoder consists of 6 Layers with 2 sublayers each. The newly developed self-attention in the first sublayer allows a transformer model to process all input words at once and model the relationships between all words in a sentence. This allows transformers to model long-range dependencies in a sentence faster than RNN and CNN based models. 
  - BERT
- Overview over important NLP models
  ![image](https://user-images.githubusercontent.com/16402963/119276069-bb0d1100-bbe6-11eb-9184-82408c2e74e8.png)
  
## Chapter 7 Transfer Learning for NLP I
In the first “wave” of transfer learning, ELMo (Embeddings from Language Models), ULMFiT (Universal Language Model Fine-tuning for Text Classification), and GPT (Generative Pretraining) were published. These models enhanced the performance and introduced ideas like contextual embeddings, fine-tuning, and transformers in natural language processing.
- Sequential Inductive Transfer Learning
  - Type of transfer learning:
    ![image](https://user-images.githubusercontent.com/16402963/119278856-f95dfc80-bbf5-11eb-8fe8-c075d76e7f06.png)
  - Feature Extraction vs. Fine-tuning
    Sequential inductive transfer learning mainly consists of two steps: pretraining and adoption.
    ![image](https://user-images.githubusercontent.com/16402963/119278888-2b6f5e80-bbf6-11eb-8f3f-bdc158904b07.png)
    adoption on the target task. Here, the main distinction is if the pretrained model weights are kept (embedding or feature extraction) or adjusted to the target task (fine-tuning) (Peters, Ruder, and Smith 2019)
    - In feature extraction, In the adoption phase, the LM’s weights do not change, and just the top layer of the model is used. The adopted model learns a linear combination of the top layer (Peters, Ruder, and Smith 2019).
    - fine-tuning adjusts the pretrained model’s weights on a specific task. The method’s disadvantage is that the general knowledge and relationship between words can get lost in the adjustment phase. This is called “catastrophic forgetting”. Techniques for preventing catastrophic forgetting are freezing learning rates and regularization, mainly explained in the ULMFiT section.
- Models
  - [ELMo](https://allennlp.org/elmo) (Embeddings from Language Models) - The “new age” of embeddings: 
    The Allen Institute introduced with ELMo a state-of-the-art model that can distinguish between the same word’s varying meaning in different contexts (contextual word embedding).
    - Its most significant advance compared to previous models like word2vec and Glove (chapter 3), is that ELMo can handle the different meanings of a word in different contexts (polysemy).
    - ![image](https://user-images.githubusercontent.com/16402963/119279139-ba30ab00-bbf7-11eb-8e95-ee59077e13e5.png)
    - In the adoption phase, a linear combination of the internal states is learned in an additional layer. This representation is different for every task. ELMo adopts the target by feature extraction; the pretrained model is frozen (all learning rates are set to 0), and the task-specific ELMo representations are calculated.
    - ![image](https://user-images.githubusercontent.com/16402963/119279305-cd904600-bbf8-11eb-974f-1a48732fd547.png)
    - ELMo saves semantic - context-dependent features - information more in the higher layer and syntactic information more in the lower layers. That is also why the weighting in every task should be different.
  - ULMFiT - cutting-edge model using LSTMs: 
    Howard and Ruder introduced ULMFiT, an adaption of fine-tuning in NLP.
    - ![image](https://user-images.githubusercontent.com/16402963/119279376-54452300-bbf9-11eb-83f1-fc24a3512d25.png)
    - ULMFiT follows three steps to achieve its notable transfer learning results:
      ![image](https://user-images.githubusercontent.com/16402963/119279424-b00fac00-bbf9-11eb-990d-b73413daf25d.png)   
      - Language Model pretraining
      - Language Model fine-tuning (Adoption I)
        - Discriminative fine-tuning
        - slanted triangular learning rates (STLR)  
      - Classifier fine-tuning (Adoption II)
        - Concat pooling
        - Gradual unfreezing
    - tutorial links [1](https://www.analyticsvidhya.com/blog/2018/11/tutorial-text-classification-ulmfit-fastai-library/)[2](https://medium.com/technonerds/using-fastais-ulmfit-to-make-a-state-of-the-art-multi-label-text-classifier-bf54e2943e83)[3](https://humboldt-wi.github.io/blog/research/information_systems_1819/group4_ulmfit/)       
  - GPT - First step towards transformers: 
    Open AI published GPT, the first transformer model, which is an urge for the upcoming chapters. This model architecture is mainly used in the most recent developments. To sum up, GPT is a multi-layer transformer-decoder with task-aware input transformations GPT is trained on the Book-Corpus data set, where the input is tokenized as a sub-word. 
    ![image](https://user-images.githubusercontent.com/16402963/119279756-c28ae500-bbfb-11eb-89ce-cbc1bd7215e4.png)
    - Difference compare with above model:
      - ELMo is based on word embeddings, whereas GPT is based on fine-tuning like ULMFiT.
      - GPT uses a different model architecture. Instead of the multi-layer LSTM, GPT is a multi-layer transformer decoder.
      - In contrast to ELMo, that works character-wise, GPT uses tokens (subwords) from the words.
      - GPT is trained on a large number of data (especially the subsequent models like GPT-2 or GPT-3)

## Chapter 8 Attention and Self-Attention for NLP
- Attention
  - Bahdanau-Attention
  - Luong-Attention 
- Self-Attention

## Chapter 9 Transfer Learning for NLP II
Unsupervised representation learning has been highly successful in NLP. Typically, these methods first pre-train neural networks on large-scale unlabeled text corpora and then fine-tune the models on downstream tasks. [Huggingface Transformer Github](https://github.com/huggingface/transformers)
- Bidirectional Encoder Representations from Transformers (BERT) -- Auto Encoding (AE)
  BERT is published by researchers at Google AI in 2018. It is regarded as a milestone in the NLP community by proposing a bidirectional Language model based on Transformer.
  - BERT is a notable example of AE and it uses the structure of the AutoEncoding model, which means that some tokens in the training data will be masked. BERT uses the Transformer Encoder as the structure of the pre-train model and addresses the unidirectional constraints by proposing new pre-training objectives: the Masked Language Model(MLM) and Next-sentence Prediction(NSP).
    - Masked Language Model
      - 80% of the time: Replace the word with the [MASK] token, e.g., my dog is hairy  ⇒ my dog is [MASK]
      - 10% of the time: Replace the word with a random word,e.g., my dog is hairy  ⇒my dog is apple.
      - 10% of the time: Keep the word unchanged, e.g., my dog is hairy  ⇒ my dog is hairy. (Since we keep 10% of the masked token unchanged, if we do not add random noise, the model will be “lazy” in our training, the model will plagiarize current tokens, rather than learning)
    - Next-sentence Prediction 
    ![image](https://user-images.githubusercontent.com/16402963/119280466-8f4a5500-bbff-11eb-8fe1-5253e6b0a279.png)
  - BERT limits the length of the entire sequence to no more than 512 tokens. Whether it is a one-sentence sequence or a sentence-pairs sequence, sequences exceeding 512 will be divided at intervals of 512 tokens. In practice, considering computational efficiency, BERT mostly divides the sequence with a length of 128 tokens.
  - BERT performs fine-tuning by comparing the loss between the prediction and the true value. It cannot generate new vocabulary by itself, but can only choose from the vocabulary of the latter sentence.
  - As for the Question Answering Task (e.g.Reading comprehension, task(c)), BERT needs to find the correct answer in the latter sentence to answer the question raised by the previous sentence. For each token in the second sentence, BERT will use the output embeddings of the token to make two predictions, representing whether the token is the beginning or the end of the answer.
  - [Hands on Example](https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)
- Generative Pre-Training(GPT-2) -- Auto-regressive Language Model(AR) 
  - GPT-2 is a unidirectional language model, such model structure is also called Auto-regressive language model.
    ![image](https://user-images.githubusercontent.com/16402963/119354185-c3f6f480-bc71-11eb-9444-293bee7a2b9c.png)
  - package: [AllenNLP](https://demo.allennlp.org/next-token-lm?text=AllenNLP%20is)
  - Decoder-only Block:
    ![image](https://user-images.githubusercontent.com/16402963/119354815-8181e780-bc72-11eb-912f-cafe7f44ec46.png)
   - The framework of GPT-2 is the combination of pre-training based on Transformer Decoder and fine-tuning based on unsupervised downstream tasks.
- XLNet -- Permutation Language Modeling(PLM)
  - Two-Stream Self-Attention
  - XLNet has built a bridge between language modeling and bidirectional models. Overall, XLNet is a generalized AR pre-training method that uses a permutation language modeling objective to combine the advantages of AR and AE methods 
  - ![image](https://user-images.githubusercontent.com/16402963/119355476-51871400-bc73-11eb-82ca-67a1d51c3c8c.png)

## Chapter 11 Resources and Benchmarks for NLP
- Metrics
  - Exact match (EM): The percentage of predictions that match any one of the answers exactly.
  - (Macro-averaged) F1 score (F1): Each answer and prediction is tokenized into words. For every answer to a given question, the overlap between the prediction and each answer is calculated and the maximum F1 is chosen. 
  - Perplexity: Perplexity is a measurement of how well a probability model predicts a sample. A low perplexity indicates the probability distribution is good at predicting the sample.
  - BLEU: BLEU (Bilingual Evaluation Understudy) is an algorithm for evaluating the quality of text which has been machine-translated from one natural language to another. Scores are calculated for individual translated segments—generally sentences—by comparing them with a set of good quality reference translations.
  - Accuracy: Accuracy is the ratio of number of correct predictions to the total number of input samples.
  - Matthews correlation coefficient: The MCC is used as a measure of quality of binary classifications. It takes true and false positives and negatives into account and is regarded as a balanced measure which can be used even if the classes are imbalanced. 

## Chapter 13 Natural Language Generation
Machine learning systems can be differentiated into two types: Discriminative and Generative. While discriminative systems like classification, regression, clustering are the more well known type, it’s the Generative systems that hold greater promise of achieving Artificial General Intelligence.
- Definition
  - Text-to-Text
    - Machine Translation : Automatically translating between various human languages
    - Text Summarization : Summarizing a (big) text document into a shorter summary/abstract.
  - Data-to-Text
    - Image Captioning : Describe the image in a short sentence.
    - Video Captioning : Generate captions that change with the video scenes.
    - Business Intelligence : Creating text summaries of data from conventional databases (e.g SQL)
  - Ideas-to-Text
    - Poetry/Song Generation : Generating a song from a few keywords or mimicking the style of a certain artist.
    - Fake News : Automatically generating news items that look credible but are not.
  - Dialog Systems (Chatbots)
    - Goal Oriented : Chatting with a computer system (agent) with a specific purpose (e.g. booking a flight)
    - Open ended conversations : When the conversation with the agent is casual chit-chat but has the components of information, emotion and human like empathy.
    - A chatbot touches upon all areas of NLP and NLG; for example in the above case, it would need QnA ability, text generation, summarization, information retrieval and classification, to name a few.
    ![image](https://user-images.githubusercontent.com/16402963/119395532-a6418380-bca1-11eb-9dc6-7fa71482f423.png)




 

