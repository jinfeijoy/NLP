# 2021 NLP Plan


2021 NLP Plan:

### Coding Project
- Sentiment Analysis
  - [IMDB Reviews](https://github.com/jinfeijoy/NLP/tree/main/kaggle_IMDB_Review) (May 20)
    - TFIDF+XGBoost / GloVe+RNN / fastai+BERT
  - [Twitter sentiment analysis](https://github.com/jinfeijoy/NLP/tree/main/kaggle_Twitter_sentiment) (June 3)
    - Binary/Multi-class Classification / GloVe+RNN / fastai + AWD_LSTM 
  - [Covid Vaccines tweet (Visualization)](https://github.com/jinfeijoy/NLP/tree/main/kaggle_Covid19_vaccine_Twitter) (June 11)
    - fastai + pre-trained model / Visualization Report
- Text Generation
  - [English Text Generation](https://github.com/jinfeijoy/NLP/tree/main/text_generation) (July14 - July24) 
    - fastai + AWD_LSTM / GPT2 / RNN + LSTM (from scratch)
  - [Chinese Text Generation](https://github.com/jinfeijoy/NLP/tree/main/chinese_text_generation) (3W) (Nov15)
    - web scrabing / vocabulary generation / GPT2 
  - [Weibo Data Scraping](https://github.com/jinfeijoy/NLP/tree/main/weibo_spider)
    - weibo-spider / weibo-search
- Document Summarization
  - [Document analysis (opinion mining)](https://github.com/jinfeijoy/NLP/tree/main/kaggle_article_analysis) (3W) (June12 - July2)
    - Document  Similarity / Article Sentiment / Article Classification
    - TFIDF / word2vec / transformers / XLNet / GloVe + RNN&LSTM 
  - [topic modelling and text summary](https://github.com/jinfeijoy/NLP/tree/main/topic_modelling_text_summary) (2W) (July1 - July13)
    - Text Classification / Text Summary / Document Clustering 
    - Transformers (BERT) / doc2vec + XGBoost / LSA / sumy / Similarity Matrix / Page Rank / SentenceTransformer   
- Semantic Analysis
  - [NER/relation Extraction](https://github.com/jinfeijoy/NLP/tree/main/information_extraction) (July28 - Aug26)
    - NER: BERT / Bi-LSTM / CRF / NERDA Fine-tune / NLTK / Spacy
    - Relation: networkx / stanza
  - [Key word extraction / intent classification](https://github.com/jinfeijoy/NLP/tree/main/semantic_analysis) (July22 - July26)
    - TFIDF / Text Rank / Topic Rank / YAKE / BERT
- [Recommender System](https://github.com/jinfeijoy/NLP/tree/main/recomend_system) 
  - English: Latent factor collaborative / content based recommender / nearest neighbor collaborative / hybrid recommendation (need to tweak code)
  - Chinese: BERT Classification / sentiment analysis / SnowNLP / cnsenti /text processing / word cloud / content-based-recommendation / TFIDF / LDA
- Applications
  - [Market Inteligence Monitoring](https://github.com/jinfeijoy/NLP/tree/main/market_inteligence_monitoring) (3W) (Nov27): Scraping news from web / text similarity / LDA
  - [Auto ML](https://github.com/jinfeijoy/NLP/tree/main/autoML)


### Learning Project
- [2021 NLP Trend](https://www.analyticsinsight.net/top-10-natural-language-processing-nlp-trends-for-2021/)
- Supervised learning and unsupervised learning collaboration (1W May24): 
  - [Semi-Supervised Learning](https://www.statworx.com/at/blog/5-types-of-machine-learning-algorithms-with-use-cases/#h-4-semi-supervised-learning): The objective is to learn the structure of a language in a first step before specializing in a particular task
  - [Combining supervised learning and unsupervised learning to improve word vectors](https://towardsdatascience.com/combining-supervised-learning-and-unsupervised-learning-to-improve-word-vectors-d4dea84ec36b)
  - [Semi-Supervised Learning](https://algorithmia.com/blog/semi-supervised-learning)
- Accelerating the training of large language models (July27)
  - [Determined: Deep Learning Training Platform](https://www.determined.ai/blog/faster-nlp-with-deep-learning-distributed-training): AWS
  - [ONNX Runtime (ORT)](https://www.onnxruntime.ai/): Azure
  - Google Cloud: GPU/TPU
- Auto ML (automation in NLP) (1W) (Aug24 - Aug29): 
  - [Automl In Towards Data Science](https://towardsdatascience.com/tagged/automl)
  - [What is automated machine learning (AutoML)](https://docs.microsoft.com/en-us/azure/machine-learning/concept-automated-ml) 
  - implement AutoML on Cloud (ner, structure dataset) (24-26)
  - H2O (28-30)
  - ludwig-ai (open-source auto-ml) (Dec 10)
- [Question&Answer](https://github.com/jinfeijoy/NLP/tree/main/chatbot)


### Reference:
- Text Frequency Plot:
  - https://www.kaggle.com/psbots/customer-support-meets-spacy-universe 
- Recommender System: 
  - https://www.kaggle.com/gspmoreira/recommender-systems-in-python-101
  - https://www.kaggle.com/rounakbanik/the-movies-dataset
- Kaggle Cord Research Engine:
  - https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge
  - https://www.kaggle.com/danielwolffram/topic-modeling-finding-related-articles
  - https://www.kaggle.com/dgunning/cord-research-engine-search-and-similarity
- Sentiment analysis:
  - Youtube Trending data: https://www.kaggle.com/rsrishav/youtube-trending-video-dataset?select=CA_youtube_trending_data.csv
  - Basic Sentiment analysis data: https://www.kaggle.com/kazanova/sentiment140 
  - Continuous target variable: 
    - https://www.kaggle.com/edenbd/150k-lyrics-labeled-with-spotify-valence
    - https://www.kaggle.com/wjia26/big-tech-companies-tweet-sentiment
  - Time series: https://www.kaggle.com/omermetinn/tweets-about-the-top-companies-from-2015-to-2020?select=Tweet.csv
- Opinion Mining: 
  - Covid vaccination Tweet: https://www.kaggle.com/gpreda/all-covid19-vaccines-tweets
  - Opinion Mining Application:
    - social mdeia analysis: opinion about brands and products offered on social media
    - brand awareness: public sentiment of your brands
    - customer feedback: 
    - customer service: to know how the customers feel about the interactions with employee
    - market research: to find out if there is room for new products or if certain services are falling out of favor.
    - evaluating market campaigns: 


Next Year Plan:
- Kownledge Graph
- Time Series
- Image Recognation 
- Block Chain and Data Lake
