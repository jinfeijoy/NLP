# Autoencoder
* unsupervised way to build neural network
* Different autoencoders:
    * sparse autoencoder: sparsity in h
    * denoising autoencoder: corrupt the input sample x
    * stacked autoencoder: layer-wise pretraining

* Applications:
    * learn phenotypic patterns from EHRs without expert knowledge 
        1. gaussian process regression to impute missing data
        2. two-layer sparse autoencoder (30-day window as input -> sparse autoencoder with 2 level hidden layer h1 and h2 )
        3. accurate classification can be achieved using hidden layers from autoencoder 
    * unsupervised representations of patients for general predictive healthcare  
## Reference
* https://www.jeremyjordan.me/autoencoders/
* https://www.coursera.org/learn/deep-learning-methods-healthcare/home/week/4
