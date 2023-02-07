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

