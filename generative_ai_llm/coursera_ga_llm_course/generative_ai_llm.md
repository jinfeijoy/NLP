* https://www.coursera.org/programs/manulife-learning-program-zgh8l/learn/generative-ai-with-llms
* https://github.com/Ryota-Kawamura/Generative-AI-with-LLMs

## week1 
* topics: transformer architecture 
* labs: [different prompt, different inference parameters/sampling](https://www.coursera.org/learn/generative-ai-with-llms/lecture/wno7h/lab-1-walkthrough)
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
* labs: [different transformer model from huggingface](https://www.coursera.org/learn/generative-ai-with-llms/lecture/A6TDx/lab-2-walkthrough)
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
  * [benchmark](https://www.coursera.org/learn/generative-ai-with-llms/lecture/1OMma/benchmarks)
     * <img width="845" alt="image" src="https://github.com/user-attachments/assets/94a242f9-5357-4ea2-9dcf-3f68036b0a98">
     * GLUE: sentiment analysis, question answering
     * SuperGLUE: multi-sentence reasoning, and reading comprehension
     * MMLU: tasks that extend way beyond basic language understanding
     * BIG-bench: currently consists of 204 tasks, ranging through linguistics, childhood development, math, common sense reasoning, biology, physics, social bias, software development and more.
     * HELM: aims to improve the transparency of models, and to offer guidance on which models perform well for specific tasks.

* Parameter efficient fine-tuning (PEFT)
   * only update small subset of parameters, In some cases, just 15-20% of the original LLM weights
   * full fine-tuning: <img width="920" alt="image" src="https://github.com/user-attachments/assets/5fa49a68-cd20-4066-92ef-b3c01bce5b40">
   * PEFT: <img width="705" alt="image" src="https://github.com/user-attachments/assets/39451614-8dd7-439c-88ad-554120e7f1d6">
      * trade-off: parameter efficiency, training speed, inference cost, memory effficiency, model performance
      * method: <img width="899" alt="image" src="https://github.com/user-attachments/assets/c2652d06-c87b-4c4d-ae99-e32ba4782f29">
   * reparameterization
      * Reparameterization methods also work with the original LLM parameters, but reduce the number of parameters to train by creating new low rank transformations of the original network weights
      *  LoRA: <img width="903" alt="image" src="https://github.com/user-attachments/assets/3dfc169d-b907-42fa-8a6a-a61cc77e2136">
         <img width="812" alt="image" src="https://github.com/user-attachments/assets/1a88e2a5-8817-447d-aa96-741b41319f5a">
         <img width="886" alt="image" src="https://github.com/user-attachments/assets/b0508dd2-d0f8-41db-a459-a55313e14fa3">

   * additive
      * additive methods carry out fine-tuning by keeping all of the original LLM weights frozen and introducing new trainable components
      * Adapter methods add new trainable layers to the architecture of the model, typically inside the encoder or decoder components after the attention or feed-forward layers.
      * Soft prompt methods, on the other hand, keep the model architecture fixed and frozen, and focus on manipulating the input to achieve better performance. work well in large model
         * <img width="897" alt="image" src="https://github.com/user-attachments/assets/ec608d9d-68c9-43d4-9940-14a6233959d5">
         * <img width="906" alt="image" src="https://github.com/user-attachments/assets/aff5e1f3-4b20-4172-af21-2afb30448900">



## week3 
* topics: align llm with human values to reduce error
* labs: re-inforcement learning
* reinforcement learning with human feedback
   * objective: maximum helpfulness relevance, minimize harm, avoid dangerous topics
   * Reinforcement learning is a type of machine learning in which an agent learns to make decisions related to a specific goal by taking actions in an environment, with the objective of maximizing some notion of a cumulative reward.
   * <img width="925" alt="image" src="https://github.com/user-attachments/assets/368b919d-38d0-4dc6-bc3f-4f82c517f221">
   * steps:
      * prepare dataset for human feedback
        <img width="916" alt="image" src="https://github.com/user-attachments/assets/ca7c2bdc-12ca-4b27-8a19-a8a8db61e91d">
      * collect human feedback: define your model alignment criteria; for the prompt respons sets that you just generated, obtain human feedback through labeler workforce
      * prepare labeled data for training
        <img width="949" alt="image" src="https://github.com/user-attachments/assets/a5ca8b26-0b1d-4e8b-be0b-d34770bd3c4c">
      * train reward model (supervise learning, can use bert)
        <img width="907" alt="image" src="https://github.com/user-attachments/assets/6aa75910-5012-4aaa-8f87-2b7fa2c1c7e4">
      * use reward model
        <img width="896" alt="image" src="https://github.com/user-attachments/assets/da91681a-1c9d-47fd-bcae-1e04f4ad5e99">
   * RLHF: Fine-tuning with reinforcement learning
      * This is the algorithm that takes the output of the reward model and uses it to update the LLM model weights so that the reward score increases over time. 
      * <img width="746" alt="image" src="https://github.com/user-attachments/assets/52ac66e2-28b9-4f91-a60b-ef9b587afa5c">
      * popular RL algorithm method: proximal policy optimization (ppo)
   * PPO (Proximal policy optimization)
      * The goal is to update the policy so that the reward is maximized.
      * You start PPO with your initial instruct LLM, then at a high level, each cycle of PPO goes over two phases
         * In Phase I, the LLM, is used to carry out a number of experiments, completing the given prompts. These experiments allow you to update the LLM against the reward model in Phase II.The expected reward of a completion is an important quantity used in the PPO objective. We estimate this quantity through a separate head of the LLM called the value function. The value loss makes estimates for future rewards more accurate. The value function is then used in Advantage Estimation in Phase 2
         * In Phase 2, you make a small updates to the model and evaluate the impact of those updates on your alignment goal for the model. The model weights updates are guided by the prompt completion, losses, and rewards. PPO also ensures to keep the model updates within a certain small region called the trust region. This is where the proximal aspect of PPO comes into play. Ideally, this series of small updates will move the model towards higher rewards. The PPO policy objective is the main ingredient of this method.
         * value loss: <img width="912" alt="image" src="https://github.com/user-attachments/assets/1ffffcdd-eb85-4120-84d4-03fb982f8f79">
         * policy loss: <img width="895" alt="image" src="https://github.com/user-attachments/assets/08a3f3f7-851c-409f-8532-7fbe12304038">
           <img width="910" alt="image" src="https://github.com/user-attachments/assets/b3c6d1ab-0270-44bf-be68-4f032837c5be">
         * entropy loss: <img width="893" alt="image" src="https://github.com/user-attachments/assets/0b9ea809-5afa-49e5-82d3-1abfea873302"> (similar as temprature setting, higher entropy has more creativity generation)
         * overall:  <img width="837" alt="image" src="https://github.com/user-attachments/assets/1fbeb386-fc26-4b5f-b285-8b854415e5df">
   * reward hacking
      * e.g. to avoid toxic results, by having rl，it may change the results to opposite way.
      * avoid reward hacking: to aoivd it, set the initial instruct llm as reference model, froze it and use it to compare with the rl results.  <img width="921" alt="image" src="https://github.com/user-attachments/assets/7576f9f8-8ddc-4993-b1be-996e2c6f8245">
         * [KL divergence](https://www.coursera.org/learn/generative-ai-with-llms/supplement/JESIK/kl-divergence)
  * constitutional ai: <img width="925" alt="image" src="https://github.com/user-attachments/assets/663f540f-4b8e-4c8e-a60b-d3288af9c587">

* LLM-powered application
  * optimization techniques:
    ![image](https://github.com/user-attachments/assets/d4005f2c-f02f-467d-a8bb-93a14444795c)
      * Distillation: having a larger teacher model train a smaller student model. The student model learns to statistically mimic the behavior of the teacher model, either just in the final prediction layer or in the model's hidden layers as well. You start with your fine tune LLM as your teacher model and create a smaller LLM for your student model. You freeze the teacher model's weights and use it to generate completions for your training data. At the same time, you generate completions for the training data using your student model. The knowledge distillation between teacher and student model is achieved by minimizing a loss function called the distillation loss. To calculate this loss, distillation uses the probability distribution over tokens that is produced by the teacher model's softmax layer. Now, the teacher model is already fine tuned on the training data. So the probability distribution likely closely matches the ground truth data and won't have much variation in tokens. That's why Distillation applies a little trick adding a temperature parameter to the softmax function. better for encoder only model like Burt. 
        ![image](https://github.com/user-attachments/assets/02775246-adcb-4926-9227-031233c5a34d)
      * quantization:
        ![image](https://github.com/user-attachments/assets/ff0d8e40-1995-423d-af6c-c78ab66e0511)
      * pruning
        ![image](https://github.com/user-attachments/assets/9dd8fc0f-156f-4862-96f3-84c75fc7279f)
  * project lifecycle cheat sheet 
    ![image](https://github.com/user-attachments/assets/5ac25d75-818e-4e6a-b58d-e77333a99480)
  *  
* ongoing research

  
