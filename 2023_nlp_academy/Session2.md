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


### Decoder-only (GPT)
    
### Encoder-decoder (T5)

    
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
    
## monoBERT: BERT as a reranker

## Evaluating search engines

## How to deal with long documents

* Why do we need to split documents into shorter segments?
  * models are finetuned on shorter segments
  * quadratic memory cost
* When using BM25, is there a problem when the index contains only a few documents?
  * poor IDF estimations: better to use a larger index than multiple small ones 
