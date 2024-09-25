## Week1 summarize dialogue 

* load pre-trained model:
    ``model_name='google/flan-t5-base'``
    ``model = AutoModelForSeq2SeqLM.from_pretrained(model_name)``
* define tokenizer:  ``tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)``
* encode:
  * no prompt: ``sentence_encoded = tokenizer(sentence, return_tensors='pt')``
  *  Zero Shot Inference with an Instruction Prompt:
      ``prompt = f"""
        Summarize the following conversation.
        {dialogue}
        Summary:
            """
        tokenizer(prompt, return_tensors='pt')``
  * Zero Shot Inference with the Prompt Template from FLAN-T5
      ``prompt = f"""
        Dialogue:
        {dialogue}
        What was going on?
            """
        tokenizer(prompt, return_tensors='pt')``
  * One Shot Inference:
      ![image](https://github.com/user-attachments/assets/a95a8740-a399-4413-b347-b924b1bc23c6)
      ``one_shot_prompt = make_prompt(example_indices_full, example_index_to_summarize) # example_indices_full = [40], example_index_to_summarize = 100
        inputs = tokenizer(one_shot_prompt, return_tensors='pt')``
  * Few Shot Inference
      ``few_shot_prompt  = make_prompt(example_indices_full, example_index_to_summarize) # example_indices_full = [40,50,60], example_index_to_summarize = 100
          inputs = tokenizer(few_shot_prompt , return_tensors='pt')``
       
* decode:
  * original decode:  ``sentence_decoded = tokenizer.decode( sentence_encoded["input_ids"][0],  skip_special_tokens=True )``
  * summarization (generate new word): ``tokenizer.decode(model.generate(inputs["input_ids"],max_new_tokens=50,)[0],skip_special_tokens=True )``
  * Generative Configuration Parameters for Inference: ``generation_config = GenerationConfig(max_new_tokens=50, do_sample=True, temperature=0.5)
    output = tokenizer.decode(
          model.generate(
              inputs["input_ids"],
              generation_config=generation_config,
          )[0], 
          skip_special_tokens=True
      )``
        * Choosing max_new_tokens=10 will make the output text too short, so the dialogue summary will be cut.
        * Putting do_sample = True and changing the temperature value you get more flexibility in the output.

## Week2

## [Week3](https://www.coursera.org/learn/generative-ai-with-llms/lecture/yPzI4/lab-3-walkthrough)
* package:
  * ``peft``: PEFT, ``trl``: PPO
  * ``AutoModelForSeq1Classification`` in transformer: load Facebook binary classifier, seq1 classifier. Which when we give it a string of text or a sequence of text, it'll tell us whether or not that text contains hate speech or not, with a particular distribution across not hate or hate
  * ![image](https://github.com/user-attachments/assets/9b10de10-b04b-48e7-88dc-edcb958dbae7)
* steps:
    * prepare dataset: tokenize, add prompt
    * load model parameters
    * fine-tune peft:
      ![image](https://github.com/user-attachments/assets/a0b0d1dc-7e67-43f3-b524-f7c7d55472fc)
    * ppo model:
      ![image](https://github.com/user-attachments/assets/6f8ae376-1007-4344-b7c9-1d3dee65f3ea)
        * ``is_trainable=TRUE``: put the model into fine-tuning mode , when do prediction or generate summary, put it as ``FALSE``
    * reference model (output from week2 lab, input in week3 lab): original model not to fine-tune, then KL divergence is used to compare what would the original model have generated versus what would the current PPO model have generated, and then keeps things sort of in line that way and then minimizes the model's ability to perform the reward hacking. 
    * prepare reward model:
      ![image](https://github.com/user-attachments/assets/1ec9c726-3a00-44fa-ad53-bfbf4705f5e9)
      ![image](https://github.com/user-attachments/assets/65d23bc5-9ec7-4445-82f5-458569234e64)
    * create pipeline:
      ![image](https://github.com/user-attachments/assets/65b7762c-9036-4b75-8cd5-0f8084ba3765)
    * evaluate toxicity
      ![image](https://github.com/user-attachments/assets/a02e422b-23b5-4d2e-b9e5-0182ad4a53e8)
      ![image](https://github.com/user-attachments/assets/a583d6c9-c628-40ed-b4c6-6f87a0c51ffc)
      ![image](https://github.com/user-attachments/assets/7edccb0b-2895-4793-a1cc-7a8e1f773bc3)
      ![image](https://github.com/user-attachments/assets/e762bf42-9ec1-4e39-ac8e-970f90f555e3)
    * perform fine-tune -- initialize PPOTrainer
      ![image](https://github.com/user-attachments/assets/ddec9a2d-c23c-439e-bb75-4a2786038089)
    * perform fine-tune -- fine-tune the model:
        * objective/kl: minimize kl divergence
        * ppo/returns/mean: maximize mean returns
        * ppo/policy/advantages_mean: maximize advantages
        * ![image](https://github.com/user-attachments/assets/14a1a9e3-f781-4db7-a206-f3de161b2856)
        * steps:
            * get all samples
            * summarize text
            * using sentiment pipeline to classify query response
            * pull out non-hate response and pass them into ppo trainer
            * ppo trainer to minimize loss function
    * evaluate model 


      

    * 

    * 


