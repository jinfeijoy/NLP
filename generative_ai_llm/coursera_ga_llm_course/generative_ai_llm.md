* https://www.coursera.org/programs/manulife-learning-program-zgh8l/learn/generative-ai-with-llms
* https://github.com/Ryota-Kawamura/Generative-AI-with-LLMs

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
  * different models:
    * <img width="898" alt="image" src="https://github.com/user-attachments/assets/7eb4fbf6-0c2b-4573-8120-0b348947a8d2">
    * encoder only (autoencoding)
      * <img width="904" alt="image" src="https://github.com/user-attachments/assets/f6c06158-996c-4ffd-a20b-3ec1b5f0115a">
      * (MLM masked language modelling): objective reconstruct text ('denoising')
      * (``BERT, ROBERTA``)
      * sentiment analysis, ner, word classification
    * decoder only (autoregressive)
      * <img width="918" alt="image" src="https://github.com/user-attachments/assets/24a3ee78-60f7-4a11-ace6-e5446a38c0f1">
      * (CLM causal language modelling): objective to predict next token
      * (``GPT, BLOOM``)
      * text generation, other emergent behaviour (depend on model size)
    * encoder-decoder (sequence to sequence)
      *  <img width="913" alt="image" src="https://github.com/user-attachments/assets/a9e867e6-c2ce-49e0-8a04-6b6467339962">
      * (``BART, T5``)
      * text summarization, translation, question answering 
  * adapt & align models: 0/1/few shot inference (no more than 5 or 6 shot) ; fine-tuning ; reinforcement learning 

## week2 
* topics: adapting pre-trained model to specific task
* labs: different transformer model from huggingface
* Fine-tuning LLMs with instruction 

  * pre-training is unsupervised learning, fine-tuning is supervised learning to use the data with labeled examples to update weight for LLM, the labeled examples are prompt-completion pairs
     * ![image](https://github.com/user-attachments/assets/632a0e6e-d189-470a-8af1-b144fb3b7846)
     * using prompts to fine-tune LLM with instructiton: full fine-tuning updates all parameters to improve performance
     * ![image](https://github.com/user-attachments/assets/199ee342-81c5-44f3-92bd-8d1beb1a73af)
  * steps:
     * prepare training dataset
        * sample prompt instruction templates:
           * ![image](https://github.com/user-attachments/assets/b6b714f1-0e11-42f2-b724-65b3ea52edaa)
           * Prompt template libraries for different tasks and dataset  
     * divide data into training/validation/testing
        * ![image](https://github.com/user-attachments/assets/de2898eb-d088-4372-b920-fe46a28057e7)
     * pass data to the model, get result, compare with label and calculate loss, backprobagation to update weight, for n epocs
        * ![image](https://github.com/user-attachments/assets/9b543c4a-871f-467f-9efa-4bcb49390ec1)
        * you select prompts from your training data set and pass them to the LLM, which then generates completions. Next, you compare the LLM completion with the response specified in the training data. You can see here that the model didn't do a great job, it classified the review as neutral, which is a bit of an understatement. The review is clearly very positive. Remember that the output of an LLM is a probability distribution across tokens. So you can compare the distribution of the completion and that of the training label and use the standard crossentropy function to calculate loss between the two token distributions. And then use the calculated loss to update your model weights in standard backpropagation. You'll do this for many batches of prompt completion pairs and over several epochs, update the weights so that the model's performance on the task improves
     * evaluation in validation/testing dataset 
  * fine-tuning on a single task
     * often, only 500-1000 examples needed to fine-tune a single task 
     * drawback: catastrophic forgetting (overfitting, learn new task but forget pre-trained model to generate general words), solutions to resolve this:
        * multi-task instruction fine-tuning
        * Parameter efficient fine-tuning (PEFT)
  * multi-task instruction fine-tuning
     * require more data, require 50000-100000 examples 
     * ![image](https://github.com/user-attachments/assets/0f6a9e39-2b6f-4bb2-876f-f738f0c48309)
     * [FLAN-T5](https://www.coursera.org/learn/generative-ai-with-llms/supplement/aDQwy/scaling-instruct-models) (example, fine-tune on T5)
        * ![image](https://github.com/user-attachments/assets/7c877390-be1a-4601-9085-e1001c296ffd)
  * model evaluation
     *  ROUGE: or recall oriented under study for jesting evaluation is primarily employed to assess the quality of automatically generated summaries by comparing them to human-generated reference summaries
        * ![image](https://github.com/user-attachments/assets/c7f6fa9f-4930-4a07-b22c-74ad664121f2) (can change ROUGE-1 to ROUGE-2 by changing from unigram to bi-gram)
        * ![image](https://github.com/user-attachments/assets/5f40ea5d-b659-4576-96d5-035db7f02047)
        * ![image](https://github.com/user-attachments/assets/1429d202-b105-4e8c-85ac-0da60b14a6ce)
        * ROUGE-L (longest common subsequence): ![image](https://github.com/user-attachments/assets/3ebacf3c-e754-419d-87d4-7d15058880e8)
        * ROUGE clipping: ![image](https://github.com/user-attachments/assets/80eb18bd-3348-4ee1-b081-36d593c54009)
     *  BLEU SCORE: bilingual evaluation understudy is an algorithm designed to evaluate the quality of machine-translated text, again, by comparing it to human-generated translations
        * ![image](https://github.com/user-attachments/assets/4d0bce59-df10-45c1-af2e-e4cac98d305d)
  









## week3 
* topics: align llm with human values to reduce error
* labs: re-inforcement learning 
