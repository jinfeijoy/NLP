## Dataset:
* [Covie Article Sentiment](https://www.kaggle.com/saurabhshahane/covid-19-online-articles) (June12 - June19)
    - TFIDF + XGBoost
    - word embedding RNN (tfidf+xgboost performance is better than word-embedding + rnn/lstm)
    - AWD_LSTM
    - contextual word embedding
    - transformer
        - covid_article_fastai1_with_transformer_googlecolab.ipynb was first try, need to be modified to optimize memory usage.
 
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
