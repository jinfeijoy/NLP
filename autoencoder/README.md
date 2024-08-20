# Autoencoder
* unsupervised way to build neural network
* [Different autoencoders](https://d3c33hcgiwev3.cloudfront.net/RMlXEFh3SYiJVxBYdxmI3g_23bb551921654bf7aa813a07cc64982f_lec8-AE.pdf?Expires=1724284800&Signature=BZclJ-FhFTtxMZoxca0LC6bbW2pg8AsiMWZfg7Ejh4KufEexxQF1fsKMDui63rBuGGjp4-GqtRoxEAs~z1~y3l5dKfku8DZYIGsBZFEjHTiBs~juBPfNVbRZf31cZr819vdIRH2K0RJHeTpWduUG6wCRMYnYzp87OFv9oJDzFyI_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A):
    * sparse autoencoder: sparsity in h
    * ![image](https://github.com/user-attachments/assets/1db22683-da91-4b99-a92f-0d49c8c399f4)
    * denoising autoencoder: corrupt the input sample x, then minimize reconstruction error L(input,output)
    * ![image](https://github.com/user-attachments/assets/0156d624-fa04-428f-a5bf-35386ace4824)
    * stacked autoencoder: layer-wise pretraining
    * ![image](https://github.com/user-attachments/assets/d1556e50-eb54-425f-85e1-04af5694a471)


* Applications:
    * learn phenotypic patterns from EHRs without expert knowledge 
        1. gaussian process regression to impute missing data
        2. two-layer sparse autoencoder (30-day window as input -> sparse autoencoder with 2 level hidden layer h1 and h2 )
        3. accurate classification can be achieved using hidden layers from autoencoder 
    * unsupervised representations of patients for general predictive healthcare  
## Reference
* https://www.jeremyjordan.me/autoencoders/
* https://www.coursera.org/learn/deep-learning-methods-healthcare/home/week/4
