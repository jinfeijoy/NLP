# From Text Classification to Reranking

## Overview of different transformer architectures: 

* Few-shot models (e.g. GPT-3): fast development (hrs-days) but expensive in production
  * no training step, no weight change 
* Finetuned (e.g. BERT/T5): slower development (weeks-months) but lower inference costs
* Recent trend in IR: large few-shot models generate training data for smaller finetuned models (e.g. InPars and Promptagator)

### Encoder-only (BERT)
* BERT
  * ![image](https://user-images.githubusercontent.com/16402963/218223967-9b4091b1-0e4b-4f5f-a642-f91835ed616e.png)
  * Pre-training: Self-supervised: unlimited training data
  * Ffine-tuning: Supervised: 100-1000's labeled examples
  * transformer (encoder-only) with lots of parameters + lots of texts + lots of compute
* BERT Pretraining: Masked language modeling
  * ![image](https://user-images.githubusercontent.com/16402963/218224374-b7387eef-e57d-4d1b-9323-797e8df7dafa.png)
  * random masking -> token embeddings -> blackbox -> softmax to get probability of (e.g. top 3000 words in a vocabulary) (this step will look at surrounding words)
  * to identify mask word, it will not only look at past words, but also look at next words 
* BERT Finetuning
  * ![image](https://user-images.githubusercontent.com/16402963/218224953-4edd7d80-d291-4ce1-a569-26b7c0c49f71.png)
* Encoder-only
  * Bidirectional: left and right tokens can be seen, that is it has no causality as we have access to the complete input sequence at inference time
  * Good at classifying sequences or tokens (ex.NER) needs finetuning
  * Ex: BERT, ToBERTa, XLM, DeBERTa (this is a good one) 

### Decoder-only (GPT)
* Causal, i.e. right (future) tokens are 'removed' from the attention calculation
* Good text generator
* With +3B parameters, it wors as a few-shot learner
* With finetuning, it's not as good as encoder-only or encoder-decoder
* Ex: GPT-2, GPT-3, BLOOM, OPT
    
### Encoder-decoder (T5)
* Encoder+Decoder with causal attention + cross-attention (to communicate with the encoder)
* separate input vs output
* good at everything as long as you have a supervised dataset to finetune it
* BART, T5, FLAN-T5

    
## Text classification with BERT, GPT-3 and T5
* BERT
    * ![image](https://user-images.githubusercontent.com/16402963/218233817-a60d9b4e-20a4-4c5a-9e14-65c0202e5818.png) 
    * CLS token has been influenced by other words, so the CLS token will be used as input during finetuning
    * CLS token somehome capther the similar information as mean(T1,...,Tn)
    * CLS -> Linearr (D * 2) -> Softmax P(T/F|W)     
* GPT2/GPT3 (don't recommend using decoder only model for classification model) (good for generate text because of causal mask)
    * Causal mask: only left token impact right token (next token prediction)
    * Tn -> Linear (D*V) (pretrained) -> softmax (P'positive|W)
    * Tn is the final token which capture all previous token information
    * when finetuning GPT2/GPT3, underperform than BERT
* T5: Full transformation (encoder+decoder)
    * translation/classification/
    * sequence sequence model, text in and text out
    * it was trained to know when to stop (with '\<sos>')
    * ![image](https://user-images.githubusercontent.com/16402963/218234362-f54ff634-a8b5-4e97-be2e-2829a7511003.png)
    * as text classifier: Decoder input as \<sos> or \<sos> positive, and target are 2 words: "positive" and "\<eos>"
    * all encoder and decoder are pre-trained, so less examples required to train

## Finetune with reinforcement leraining
* hot topic: how to adapt model into reinforcement learning, reinforcement learning is an extra layer of supervise learning, since we don't need supervised target as input, reinforcement learning just give reward signal thing that you made correct/wrong translation. 

## monoBERT: BERT as a reranker
* models: monoBERT, monoT5, miniLM
* monoBERT: BERT reranker
  * ![image](https://user-images.githubusercontent.com/16402963/218261580-62a921b0-637f-471a-b9ae-83fede6cc58a.png)
  * output is always binary classification (relevent/non-relevent)
  * Loss is the combination of positive doc and negative doc

## Evaluating search engines
* TREC-style Polling
  * manually create 50-100 queries:
    * for each query:
      * retrive documents from one or more search system
      * manually annotate top k (e.g. 10) documents as relevant or not to the query
    * compute metrics such as nDCG@10 for each system
  * most common
  * reliable but expensive: change in your search system might require a new round of annotations/when the pool of system are large and diverse, need only one round of annotation (e.g. Robust04)

* IR Metrics
  * Benchmarking: relevance judgments (is this document relevant to the query? rel(q,d) = 0/1
  * Precision: relevant docs returned/results returned 
  * Recall: relevant docs returned / relevant docs in collection
  * Reciprocal Rank: RR(R, q) = 1/rank(i)
  * Average Precision: precision when relevant doc returned / # relevant docs in collection
  * Graded relevance judgments: how relevant is this document to the query
  * DCG: how relevant/how early
  * nDCG: DCG / ideal DCG (sorted by relevance)

## How to deal with long documents

* Why do we need to split documents into shorter segments?
  * models are finetuned on shorter segments
  * quadratic memory cost
* When using BM25, is there a problem when the index contains only a few documents?
  * poor IDF estimations: better to use a larger index than multiple small ones 
