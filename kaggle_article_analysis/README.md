## Dataset:
* [Covie Article Sentiment](https://www.kaggle.com/saurabhshahane/covid-19-online-articles) (June12 - June19)
    - TFIDF + XGBoost (0.85 accuracy)
    - word embedding RNN (tfidf+xgboost performance is better than word-embedding + rnn/lstm 0.76 accuracy)
    - AWD_LSTM (AWD_LSTM is better than tfidf+xgboost 0.883 accuracy)
    - XLNet: time consuming, run with val accuracy as 0.928 with Epoch1
    - [transformer](https://github.com/jinfeijoy/NLP/blob/main/kaggle_article_analysis/Notebook/covid19_article_sentiment/Covid19_article_sentiment_transformers.ipynb): can use multiple different transformers, run "bert-base-cased" successfully on sample dataset; this code can be used as templete
        - covid_article_fastai1_with_transformer_googlecolab.ipynb was first try, need to be modified to optimize memory usage, failed
 
* [Stock News Sentiment](https://www.kaggle.com/sidarcidiacono/news-sentiment-analysis-for-stock-data-by-company)
* [Research Article Classification](https://www.kaggle.com/blessondensil294/topic-modeling-for-research-articles?select=train.csv)


## Tasks:
* Exploration: covid article sentiment
* Sentiment analsis:stock news sentiment
* Topic classification: research article classification


## Key words:
* Doc2Vect
* Transformers
* Text similarity

## Notebook
* document similarity:
    * document_similarity.ipynb
        * data process
        * word cloud
        * word frequent
        * document similarity (tfidf, word2vec, transformers)   
    * document_similarity_goclab.ipynb
        * sentiment analysis
        * Q&A    
* Covie Article Sentiment:
    * Covid19_article_wordEmbedding.ipynb
        * TFIDF+XGBoost, glove+RNN/LSTM
    * Covid19_article_wordEmbedding_gcolab.ipynb
        * AWD_LSTM   
    * Covid19_article_sentiment_XLNet.ipynb
        * XLNet 
    * Covid19_article_sentiment_transformers.ipynb
        * transformers 
