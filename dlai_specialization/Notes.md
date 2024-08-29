## Coursera NLP Specialization
 * https://www.coursera.org/specializations/natural-language-processing
 * [assignment](https://github.com/amanchadha/coursera-natural-language-processing-specialization/tree/master/2%20-%20Natural%20Language%20Processing%20with%20Probabilistic%20Models/Week%202)

 ### Course 1 Natural Language Processing with Classification and Vector Spaces
 #### Week2: Naive Bayes

 * [training naive bayese steps](https://www.coursera.org/learn/classification-vector-spaces-in-nlp/supplement/0imfM/training-naive-bayes)
    * preprocessing: ![image](https://github.com/user-attachments/assets/5723ae49-739e-4c45-830b-303fa4162564)
    * word counts ( compute freq(w,class)): ![image](https://github.com/user-attachments/assets/d50bce5f-d502-4e09-abf5-77ddbf51825a)
    * P(w|class) ; get lambda: ![image](https://github.com/user-attachments/assets/8a81ba97-c7e8-4500-8468-fffd327b65b3)
    * get the log prior = log(P(pos)/P(neg)): ![image](https://github.com/user-attachments/assets/93c6b882-f23d-4a59-804a-5e61670af7e8)
* [Error analysis](https://www.coursera.org/learn/classification-vector-spaces-in-nlp/supplement/clcHR/error-analysis):several mistakes may cause misclassifying
    * removing punctuation (e.g. removing :( may cause sentiment analysis error)
    * removing words (e.g. removing 'not')
    * word order
    * Adversarial attacks: These include sarcasm, irony, euphemisms.


#### Week3 Vector Space Model
* represent words and documents as vectors, representation that captures relative meaning 
* Vector space design
    * word by word design: number of times they occur together within a certain distance k
    * word by document design: number of times a word occurs within a certain category (topics)
* similarity: 
    * euclidean distance: ![image](https://github.com/user-attachments/assets/f42fea25-ac38-4569-8aff-c3d719c40166)
    * cosine similarity (when corpora are different sizes):  ![image](https://github.com/user-attachments/assets/f9d09256-b921-4692-89d4-5f9ec0505f52)
* [PCA](https://www.coursera.org/learn/classification-vector-spaces-in-nlp/supplement/Xd2w5/pca-algorithm) can be used to visualize high dimensional data

#### Week4 Searching Document
* ![image](https://github.com/user-attachments/assets/1ce37ff6-752b-44bc-8005-b0a68fde63e0)
* KNN: using hashing can speed up search
* [Hash](https://www.coursera.org/learn/classification-vector-spaces-in-nlp/supplement/UFnGD/hash-tables-and-hash-functions)
* [locality sensitive hashing](https://www.coursera.org/learn/classification-vector-spaces-in-nlp/supplement/ieYM6/locality-sensitive-hashing)

### Course 2 Natural Language Processing with Probabilistic Models

#### Week 1 Autocorrect and Minimum Edit Distance
* steps for autocorrect:
   * Identify a misspelled word
   * Find strings n edit distance away: (these could be random strings): insert, delete, switch, replace
   * Filter candidates: (keep only the real words from the previous steps)
   * Calculate word probabilities: (choose the word that is most likely to occur in that context)
* [Minimum edit distance](https://www.coursera.org/learn/probabilistic-models-in-nlp/supplement/3EvOV/minimum-edit-distance-algorithm-ii): dynamic programming

#### Week 2 part of speach tagging
* Part of Speech Tagging (POS) is the process of assigning a part of speech to a word
* Markov Chains: can be used to identify the probability of the next word
   * <img width="665" alt="image" src="https://github.com/user-attachments/assets/549e114b-ec2e-45d2-a7b5-e56ccea7f30d">
   * states: A state refers to a certain condition of the present moment.  You can think of these as the POS tags of the current word. Q  is the set of all states in your model. 
   * transition matrix: The transition probabilities allowed you to identify the transition probability from one POS to another.
   * <img width="741" alt="image" src="https://github.com/user-attachments/assets/4fa02a72-4994-4c30-9e59-04d7c92e9939">
* Hidden Markov Models
   * In hidden markov models you make use of emission probabilities that give you the probability to go from one state (POS tag) to a specific word.
   * <img width="793" alt="image" src="https://github.com/user-attachments/assets/34de0ee6-cda9-4174-8933-314e78d11e6d">
* [Viterbi algorithm](https://www.coursera.org/learn/probabilistic-models-in-nlp/supplement/7efbd/the-viterbi-algorithm): initialization, forward pass, backward pass

#### Week 3 autocomplete and language model
* [language model evaluation](https://www.coursera.org/learn/probabilistic-models-in-nlp/supplement/71fL4/language-model-evaluation): Perplexity, smaller, better, the target perplexity of 20 to 60.
* for unkonwn words, use ```<UNK>```
* Criteria to create the vocabulary
   * Min word frequency f
   * Max |V|, include words by frequency
   * Use ```<UNK>``` sparingly (Why?)
   * Perplexity -  only compare LMs with the same V
* smoothing method: laplacian smoothing, kneser-ney smoothing, good-turning smoothing

#### Week 4 Word embeddings and NN
* word embedding process
   *  corpus
   *  embedding method: The task is self-supervised: it is both unsupervised in the sense that the input data — the corpus — is unlabelled, and supervised in the sense that the data itself provides the necessary context which would ordinarily make up the labels.
* embedding methods (static embedding):
   * word2vec (2013): (CBOW: continuous bag-of-words / SGNS: continuous skip-gram, skip-gram with negative sampling)
   * Global vectors (2014): (GloVe)
   * fastText (2016): support out-of-vacabulary (OOV) words
* advanced word embedding method (dynamic): deep learning, contextual embeddings, tunable pre-trained models available 
   * BERT (2018)
   * ELMo (2018)
   * GPT-2 (2018)
* embedding evaluation method
   * [intrinsic](https://www.coursera.org/learn/probabilistic-models-in-nlp/lecture/BELqR/evaluating-word-embeddings-intrinsic-evaluation): Intrinsic evaluation allows you to test relationships between words. It allows you to capture semantic analogies as, “France” is to “Paris” as “Italy” is to <?> and also syntactic analogies as “seen” is to “saw” as “been” is to <?>. 
   * [extrinsic](https://www.coursera.org/learn/probabilistic-models-in-nlp/lecture/SEJkb/evaluating-word-embeddings-extrinsic-evaluation) : Extrinsic evaluation tests word embeddings on external tasks like named entity recognition, parts-of-speech tagging, etc. 


### Course 3 sequantial model

#### week 4 Siamese Network
* Siamese Networks: learns what makes two inputs the same
* use case: handwritten check, question duplicate, queries
* <img width="697" alt="image" src="https://github.com/user-attachments/assets/738e92de-d52a-4910-a0bb-5005f44d25b7">
    * These two sub-networks are sister-networks which come together to produce a similarity score. Not all Siamese networks will be designed to contain LSTMs. One thing to remember is that sub-networks share identical parameters. This means that you only need to train one set of weights and not two. 
    * given threshold, if similarity > threshold -> same, otherwise, different
* loss function:  [triplet loss](https://www.coursera.org/learn/sequence-models-in-nlp/lecture/L88EY/triplets)
