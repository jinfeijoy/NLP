## week1 machine translation
* BLEU score:
    * package: sacrebleu
    * ``tokenized_ref = nltk.word_tokenize(reference.lower())
        tokenized_cand_1 = nltk.word_tokenize(candidate_1.lower())
        bleu_score(tokenized_cand_1, tokenized_ref)``
