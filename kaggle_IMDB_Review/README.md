# kaggle_IMDB_Review

### Dataset: 
  * [Kaggle IMDB Review](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
  * Other Data source:
    - Kaggle Amazon Sentiment Analysis https://www.kaggle.com/bittlingmayer/amazonreviews


### Reference:
  - [Semi Supervised Learning](https://becominghuman.ai/an-implementation-of-semi-supervised-learning-e0054ab4fa02): (the simplest semi-supervised algorithm, namely self-learning) Train the classifier with the existing labeled dataset. Predict a portion of samples using the trained classifier. Add the predicted data with high confidentiality score into training set. Repeat all steps above.


### Code Summary:
 - Local:
   - Word Embedding (GloVe + RNN): [IMDB_Review_WordEmbedding](http://localhost:8888/notebooks/Desktop/PersonalLearning/GitHub/NLP/kaggle_IMDB_Review/notebook/IMDB_Review_WordEmbedding.ipynb)
   - IFIDF: [IMDB_Review_RandomForest](http://localhost:8888/notebooks/Desktop/PersonalLearning/GitHub/NLP/kaggle_IMDB_Review/notebook/IMDB_Review_RandomForest.ipynb)
   - TFIDF & Glove/RNN: [IMDB_Review_Participate](http://localhost:8888/notebooks/Desktop/PersonalLearning/GitHub/NLP/kaggle_IMDB_Review/notebook/IMDB_Review_Participate.ipynb)
 - Google Colab:  
   - fastai with pre-trained model + fine-tune: [IMDB_fastai1_with_transformer_googlecolab](http://localhost:8888/notebooks/Desktop/PersonalLearning/GitHub/NLP/kaggle_IMDB_Review/notebook/IMDB_fastai1_with_transformer_googlecolab.ipynb)


