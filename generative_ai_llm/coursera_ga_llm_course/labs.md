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
