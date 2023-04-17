# Introduction and background

### NLP Timeline
* pre DL (<2014): 1 specialist for each task
* pre-BERT (2014-2018): 1 model for 1 task: search/NER/QA/Summarization/translation
* BERT (2018): 1 model for all tasks with small modification in the architecture
* T5 (2019): 1 model for all tasks, only need finetuning
* GPT-3 (2020): No finetuning; training examples as input
* GPT 3.5 (2022): Zero-shot, only instructions
* ChatGPT (2022): Conversation with instructions + history

### What are LLMs (GPT3/PALM)
* Original text -> input tokens -> token embeddings -> A decoder-only Transformer with lots of parameters -> linear layer -> softmax
* Loss = -log(P"cuisine" | input)

### Model available
* Open Source
  * OPT (Meta): 125M to 175B parameters
  * Galactica (Meta): 125M to 120B
  * Bloom (BigScience) 175B (multulingual)
  * GPT-J (EleutherAI): 6B
  * GPT-NeoX (EleutherAI) 20B 
* Paid API
  * Open AI 
  * Alepha Alpha
  * AI21
  * Cohere
* model leaderboard
  * https://crfm.stanford.edu/helm/latest/

### FLAN-T5: Open-source alternative to GPT-3.5
* Reasonable zero/few-shot with the XL model
* need at a gpi with 16GB of ram

### Information Retrieval
* Given: query q, collection of texts
* Return: a ranked list of k texts
* Maximizing: a metric of interest
* A smimple Search Engine: 
  * ![image](https://user-images.githubusercontent.com/16402963/217909441-0904ee2c-2f00-409b-8562-674a66e1328a.png) 
  * text -> inverted index -> initial retrieval (e.g. BM25) -> reranker (e.g. monoBERT, monoT5, miniLM) -> ranked list
  * inverted index: a dictionary whose keys are words and values are documents that contain those words (e.g. {'apple': [doc_21, doc_5], 'house': [doc1, doc2]})
  * retrieval/ranking: for each word q in the query, compute a score for each document D that contains word (utilized IDF in score calculation)
  * suffer from the 'vocabulary mismatch problem': car and automobile are completely different to BM25
  * it's a hard-to-beat algorithm, try BM25 first before other solutions, sentence bert can also be a good option
  * (AP: average precision)
  * monoBERT: BERT reranker: a binary classifier finetuned on pairs of <query, relevant text> and <query, non-relevant text> 


### Visconde: multi-document QA with GPT-3 and Neural Reranking
* paper can be found [here](https://arxiv.org/abs/2212.09656)
* ![image](https://user-images.githubusercontent.com/16402963/217906181-dd8bd815-ef7d-4e0c-866a-52b9decd3727.png)
  * question decomposition: few-shot GPT3
  * document retrieval: BM25 + monoT5
  * aggregation: few-shot GPT3 
* informration need -> document retrieval (text from social media and online news as documents) -> relevant documents -> aggregation -> summary
