## [Modern Approach in NLP Processing](https://compstat-lmu.github.io/seminar_nlp_ss20/introduction-deep-learning-for-nlp.html)
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
- Structure and Training of Simple RNNs
- 
