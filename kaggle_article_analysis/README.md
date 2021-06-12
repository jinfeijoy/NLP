Dataset can be found here: 
- contextual string embeddings
- flair

covid_article_fastai1_with_transformer_googlecolab.ipynb was first try, need to be modified to optimize memory usage.

Document Summarization

article analysis: word embedding RNN, AWD_LSTM, transformer

document generation

Dataset:
* [Covie Article Sentiment](https://www.kaggle.com/saurabhshahane/covid-19-online-articles)
* [Covid News Sentiment](https://www.kaggle.com/databar/coronavirus-articles-marchapril-2020-with-sent)
* [Stock News Sentiment](https://www.kaggle.com/sidarcidiacono/news-sentiment-analysis-for-stock-data-by-company)
* [Research Article Classification](https://www.kaggle.com/blessondensil294/topic-modeling-for-research-articles?select=train.csv)
* [Taylor Swift Song Lyrics](https://www.kaggle.com/PromptCloudHQ/taylor-swift-song-lyrics-from-all-the-albums)

Tasks:
* Exploration: covid article sentiment
* Topic Summary: covid News
    - Summarization
      - [PEGASUS: Google’s State of the Art Abstractive Summarization Model](https://towardsdatascience.com/pegasus-google-state-of-the-art-abstractive-summarization-model-627b1bbbc5ce) 
      - [Fine-tuning BART for Abstractive Text Summarisation with fastai2](https://medium.com/curation-corporation/fine-tuning-bart-for-abstractive-text-summarisation-with-fastai2-d7a2ad676a13)
* News classification: 
