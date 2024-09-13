

## week1 
* topics: transformer architecture 
* labs: different prompt, different inference parameters/sampling
* transformer architecture
  * tokenizer -> embedding -> encoding -> positional encoding -> self attention layer (multi-head attention) for encoder -> feed forward network (fully connected) for encoder -> output of decoder
  * tokenizer -> embedding -> encoding -> positinal encoding -> with encoder output -> self attention layer for decoder -> feed forward network for decoder -> softmax output layer -> feed it into the next token and repeat the process until the model predict end sequence token
  * <img width="914" alt="image" src="https://github.com/user-attachments/assets/21cfd083-e6a7-4ac1-b5fc-a73f5afcb339">
  * https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/idP8oaT2QPK8TAdEt6WYEg_41e77abe4ea6457e9706e0c4fa379ff1_image.png?expiry=1726358400000&hmac=_INoqUSA_H91QoKtYTirVpCcbyCzdH7EhGw2FFx3EoE![image](https://github.com/user-attachments/assets/bb02b619-8f55-4191-be75-b525affcd6c7)
  * The encoder encodes input sequences into a deep representation of the structure and meaning of the input.
  * The decoder, working from input token triggers, uses the encoder's contextual understanding to generate new tokens. It does this in a loop until some stop condition has been reached.
  * Encoder-only models also work as sequence-to-sequence models, but without further modification, the input sequence and the output sequence or the same length. Their use is less common these days, but by adding additional layers to the architecture, you can train encoder-only models to perform classification tasks such as sentiment analysis, BERT is an example of an encoder-only model.
  * Encoder-decoder models, as you've seen, perform well on sequence-to-sequence tasks such as translation, where the input sequence and the output sequence can be different lengths. You can also scale and train this type of model to perform general text generation tasks. Examples of encoder-decoder models include BART as opposed to BERT and T5, the model that you'll use in the labs in this course.
  * decoder-only models are some of the most commonly used today. Again, as they have scaled, their capabilities have grown. These models can now generalize to most tasks. Popular decoder-only models include the GPT family of models, BLOOM, Jurassic, LLaMA, and many more. 
* prompt
  * zero shot inference: ask question directly
  * one shot inference: ask question with example
  * few shots inference: ask question with multiple examples (mixed examples)
* inference configuration parameters
  * max new tokens
  * sampling approach
    * greedy (default model will chose word with highest probability, not well for long sentence with repeated word) vs random sampling (random sampling from distribution)
    * top k: select an output from the top-k results after applying random-weighted strategy using the probabilities
    * top p: select an output using the random-weighted strategy with the top-ranked consecutive results by probability and with a cumulative probability <= p
  * temprature: change the shape of probability distribution of next token, higher temprature higher randomness, lower temprature lower randomness
    
* generative ai life-cycle
  * different llm: <img width="797" alt="image" src="https://github.com/user-attachments/assets/f0116d68-1a1d-4ccc-ae3a-bad7f1e8117d">
  * <img width="815" alt="image" src="https://github.com/user-attachments/assets/5fa26ba4-5ee2-4922-8c95-f1e776b2ad7f">
  * different tasks in scope: <img width="917" alt="image" src="https://github.com/user-attachments/assets/9f5665c7-0400-4bcb-a864-854e0612995f">
  * adapt & align models: 0/1/few shot inference (no more than 5 or 6 shot) ; fine-tuning ; reinforcement learning 

## week2 
* topics: adapting pre-trained model to specific task
* labs: different transformer model from huggingface


## week3 
* topics: align llm with human values to reduce error
* labs: re-inforcement learning 
