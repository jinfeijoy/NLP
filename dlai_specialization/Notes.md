## Coursera NLP Specialization
 * https://www.coursera.org/specializations/natural-language-processing 

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
