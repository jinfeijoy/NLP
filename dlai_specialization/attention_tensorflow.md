## week1 machine translation
* BLEU score:
    * package: sacrebleu
    * ``tokenized_ref = nltk.word_tokenize(reference.lower()) ``
    * ``tokenized_cand_1 = nltk.word_tokenize(candidate_1.lower())``
    * ``bleu_score(tokenized_cand_1, tokenized_ref)``
    * ![image](https://github.com/user-attachments/assets/c3a2b2d2-2162-4597-a29d-1fc989ef69e9)
* NMT model with attention
    * encoder: embedding (``tf.keras.layers.Embedding``) + rnn (``tf.keras.layers.Bidirectional``): ``rnn(embedding(context))``
    * crossAttention: multi-head attention (``tf.keras.layers.MultiHeadAttention``) + add (``tf.keras.layers.Add()`` add target and output from multi-head attention) + normalization (``tf.keras.layers.LayerNormalization()``): ``layernorm(add([target, attn_output]))``
    * decoder: embedding (``tf.keras.layers.Embedding``) + RNN before attention (``tf.keras.layers.LSTM``) + attention (``crossAttention``) + RNN after attention (``tf.keras.layers.LSTM``) + dense layer with logsoftmax activation (``tf.keras.layers.Dense``)
    * translator: encoder + decoder
* Using the model for inference
    * decoder can be used to generate next token

## week2 Transformer Summarizer
* self-attention: scaled doc product attention
* masking: padding mask (``tf.cast``), look ahead mask (``tf.linalg.band_part``)
* pre-process data: ``tf.keras.preprocessing.text.Tokenizer``, padding (``tf.keras.preprocessing.sequence.pad_sequences``)
* Full encoding
   * embedding (``tf.keras.layers.Embedding``)    
   * positional encoding: ``sin, cos``
   * encoder:
      * multiHeadAttention: ``tf.keras.layers.MultiHeadAttention``
      * Feed forward nn: 2 layers
        ``tf.keras.Sequential([
           tf.keras.layers.Dense(fully_connected_dim, activation='relu'),  # (batch_size, seq_len, d_model)
           tf.keras.layers.Dense(embedding_dim)  # (batch_size, seq_len, d_model)
       ])``
      * mha + layernorml + ffn + dropout_ffn (``tf.keras.layers.Dropout(dropout_rate)``) + layernorm2
* decoder
   * mha1 + layernorm1 + mha2 + layernorm2 + ffn + layernorm3
* transformer: encoder + decoder + final layer ``tf.keras.layers.Dense(target_vocab_size, activation='softmax')``, output: final output and attention weight
* summarizatiton
   * predict next word: transformer with (encoding padding mask + look ahead mask + decoding padding mask)
   * summarize: tokenize input with padding and expand dims + predict next word  

## week3 question answer
* preepare data
   * Pre-Training Objective: For the masked language modeling objective, a percentage of the tokenized input is randomly masked, and the model is trained to predict the original content of these masked tokens.
   * decode to natual language: tokenize text into subwords represented by integer IDs (also with function to turn numeric tokens into human readable text)
   * tokenizing and masking: tokenizes and masks input words based on a given probability
   * Creating the Pairs: pair input (text with mask) and target (mask with word)
* pre-train
   * Instantiate a new transformer model 
   * transformer +  for input and target truncating the longer sequences and padding the shorter ones with 0  (``tf.keras.preprocessing.sequence.pad_sequences``)
* fine-tune
   * Creating a list of paired question (input) and answers (target)
      * tokenize the input and the targets
      * truncate or padding input/target (``tf.keras.preprocessing.sequence.pad_sequences``)
   * Fine tune the T5 model: with given dataset and transformer, loop transformer with (encoding padding mask + look ahead mask + decoding padding mask), save updated weight 
   * Implement your Question Answering model: predict next word 


