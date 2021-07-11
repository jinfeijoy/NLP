# Text Classification (1-5)
* [Kaggle Real or Fake] https://www.kaggle.com/shivamb/real-or-fake-fake-jobposting-prediction
  * data exploration + reports items (1)
  * classification (2)
    *  transformers: 0.99 accuracy in testing dataset with bert
    *  doc2vec + multinomialnb/xgboost/logit: 0.95 accuracy in testing dataset (binary target)
  * topic modellings (job category) (4)
  * prepare reports (5)


# Text Summary (6-13)
* Topic Summary: [BBC news summary](https://www.kaggle.com/pariza/bbc-news-summary) 
    - Summarization: text_summary_overall.ipynb
      - Extraction (6-9)
        - [LSA](https://github.com/luisfredgs/LSA-Text-Summarization): text_summary_lsa.ipynb
          - package sumy 
          - [svd](https://www.youtube.com/watch?v=OIe48iAqh8E&list=LL&index=1)
        - word embedding + similarity: text_summary_glove_pagerank.ipynb
        - [CNN](https://github.com/alexvlis/extractive-document-summarization) 
      - Abstraction (10-11)
        - T5 [Transformer](https://huggingface.co/transformers/task_summary.html)
      - Reference
        -  [PEGASUS: Google’s State of the Art Abstractive Summarization Model](https://towardsdatascience.com/pegasus-google-state-of-the-art-abstractive-summarization-model-627b1bbbc5ce) 
    - topic modellings: news category (12)



# Document Generation (14-20)
* [Taylor Swift Song Lyrics](https://www.kaggle.com/PromptCloudHQ/taylor-swift-song-lyrics-from-all-the-albums)
