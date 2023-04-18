# Lab1
## Few Shot
### Openai GPT3.5 (text-davinci-003, via paid API)
~~~
!pip install openai
openai_key = ''
assert openai_key, "Please set your OpenAI API key."
import os
import openai
openai.api_key = openai_key
input_text = """ Classify the sentiment of the movie reviews below as either 'positive' or 'negative'.
              Example 1
              Moview Review: This ...
              Sentiment: Positive
              ##
              Example 2
              Moview Review: This ...
              Sentiment: Negative
              ## 
              Example 3
              Moview Review: This ...
              Sentiment
              """
response = openai.Completion.create(
  model = 'text-davinci-002',
  prompt = input_text,
  temperature = 0.0,
  max_tokens = 10,
  )
~~~
* temperature: when temperature = 0, always output the same result, while temperature > 0, the output might be differernt. In most cases the temperature can be set as 0.
### FLAN-T5 (Open-source, via free API)
~~~
!pip install huggingface-hub
from huggingface_hub.inference_api import InferenceApi

def generate_huggingface(prompt_text: str, model, max_tokens: int = 64, temperature: float = 1.0, top_p: float= 0.7):
    inference_api = InferenceApi(model)
    params = {'temperature': temperature, 'max_length': max_tokens}
    response = inference_api(prompt_text, params)
    output_text = response[0]['generated_text']
    if output_text.startswith(prompt_text):
        output_text = output_text[len(prompt_text):]
    return output_text.strip()
    
input_text = """

      """
generate_huggingface(input_text, model = 'google/flan-t5-xl', temperature = 0.0)
~~~
### GPT-Neo-2.7B (open-source, run locally)
* not good predictor for few shot case, but it's good for over 1k examples
~~~
!pip install transformers
!pip install accelerate == 0.13.2
import torch
if torch.cuda.is_available():
    dev = 'cuda:0'
else:
    dev = 'cpu'
device = torch.device(dev)

import transformers
model_name = 'EleutherAI/gpt-neo-2.7B'
model = transformer.GPTNeoForCausalLM.from_pretrained(model_name, torch_dtype = torch.float16, low_cpu_mem_usage = True)
model = model.to(device)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
input_text = """ """
input_ids = tokenizer(input_text, return_tensors = 'pt').input_ids.to(device)
gen_tokens = model.generate(
      input_ids,
      do_sample = False,
      max_length = 5,
      )
output_text = tokenizer.batch_decode(gen_tokens)[0]
print(output_text)


~~~
