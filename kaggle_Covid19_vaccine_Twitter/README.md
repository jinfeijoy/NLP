# Covid19 Vaccine Twitter Sentiment Analysis

### Dataset
- Dataset can be found here: https://www.kaggle.com/gpreda/all-covid19-vaccines-tweets

### Project Plan
  - pre-processing
  - data exploration and visualization
  - flair: https://towardsdatascience.com/text-classification-with-state-of-the-art-nlp-library-flair-b541d7add21f
  - fastai sentiment classification (pos/neg and emotion analysis based on LSTM)
  - fastai with transformer (June 4 - June 6)
    - https://github.com/morganmcg1/ntentional/blob/master/_notebooks/2020-04-24-fasthugs_language_model.ipynb
    - https://github.com/huggingface/blog/blob/master/how-to-train.md  
    - after some exploration, with transformer like BERT, the accuracy is very low, around 0.4 for sentiment target (3 category), because the twitter is too short and some tweets are even not a full sentence, so it is not worth to use transformer when we analyze tweets. 
  - opinion about different vaccines (aspect-based sentiment analysis and emotion)
  - vaccines side effect
  - vaccine update report

### Code Summary:
- Local:
  - Word Cloud + Topic Modelling: [Covid19_vaccination_twitter_exploration](http://localhost:8888/notebooks/Desktop/PersonalLearning/GitHub/NLP/kaggle_Covid19_vaccine_Twitter/notebook/Covid19_vaccination_twitter_exploration.ipynb)
  - Visualization Report: 
    - [Vaccination_tracker](http://localhost:8888/notebooks/Desktop/PersonalLearning/GitHub/NLP/kaggle_Covid19_vaccine_Twitter/notebook/Vaccination_tracker.ipynb)
    - [Covid19_vaccination_twitter_analysis](http://localhost:8888/notebooks/Desktop/PersonalLearning/GitHub/NLP/kaggle_Covid19_vaccine_Twitter/notebook/Covid19_vaccination_twitter_analysis.ipynb)
- Google Colab
  - fastai + pre-trained model + fine-tune
    - [covid_twitter_vec_fastai1_with_transformer_googlecolab](http://localhost:8888/notebooks/Desktop/PersonalLearning/GitHub/NLP/kaggle_Covid19_vaccine_Twitter/notebook/covid_twitter_vec_fastai1_with_transformer_googlecolab.ipynb)
    - [Covid_twitter_vec_sentiment_analysis_google_colab](http://localhost:8888/notebooks/Desktop/PersonalLearning/GitHub/NLP/kaggle_Covid19_vaccine_Twitter/notebook/Covid_twitter_vec_sentiment_analysis_google_colab.ipynb)
### Reference
  - [Deep Learning For NLP: Zero To Transformers & BERT](https://www.kaggle.com/tanulsingh077/deep-learning-for-nlp-zero-to-transformers-bert)
  - [Fastai with 🤗 Transformers (BERT, RoBERTa, ...)](https://www.kaggle.com/maroberti/fastai-with-transformers-bert-roberta)   
  - [fastai with roberta](https://www.kaggle.com/melissarajaram/roberta-fastai-huggingface-transformers/#data)
  - [COVID-19 Vaccine Sentiment Analysis with fastai](https://www.kaggle.com/twhelan/covid-19-vaccine-sentiment-analysis-with-fastai)
  - [NLP: Vaccine Sentiment & Tweet Generation [FastAI]](https://www.kaggle.com/joshuaswords/nlp-vaccine-sentiment-tweet-generation-fastai)
  - [Explore Vaccines Tweets](https://www.kaggle.com/gpreda/explore-vaccines-tweets)
  - [COVID-19 Vaccine Sentiment Analysis with fastai](https://www.kaggle.com/twhelan/covid-19-vaccine-sentiment-analysis-with-fastai)

### Other dataset:
- [World Vaccination Progress](https://www.kaggle.com/gpreda/covid-world-vaccination-progress)
- [Geospatial Analysis on Covid-19](https://www.kaggle.com/eswarchandt/geospatial-analysis-on-covid-19-day-to-day-track)

### Notebook:
* Covid_twitter_vec_sentiment_analysis_google_colab: run AWD_LSTM in google colab to generate emotion and sentiment for covid vacc tweets
* Covid19_vaccination_twitter_exploration: word cloud and topic modelling
* Covid19_vaccination_twitter_analysis: did analysis on the output from google_colab file, visulization tools applied
* Vaccination_tracker: track covid and covid vacc, visualization tools applied 
* covid_twitter_vec_fastai1_with_transformer_googlecolab: applied pre-trained transformer to fastai, but performance is not good (0.38), need to do further investigation in future task
