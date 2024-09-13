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
* pre-process data: ``tf.keras.preprocessing.text.Tokenizer``, padding (``tf.keras.preprocessing.sequence.pad_sequences``)
* positional encoding: ``sin, cos``
* masking: padding mask (``tf.cast``), look ahead mask (``tf.linalg.band_part``)
* self-attention: scaled doc product attention
* encoder:
   * multiHeadAttention
   * Feed forward nn: 2 layers
     ``tf.keras.Sequential([
        tf.keras.layers.Dense(fully_connected_dim, activation='relu'),  # (batch_size, seq_len, d_model)
        tf.keras.layers.Dense(embedding_dim)  # (batch_size, seq_len, d_model)
    ])``
* decoder
* transformer
* summarizatiton 



 ## week3 question answer
