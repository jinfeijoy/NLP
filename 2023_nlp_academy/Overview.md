# Outline

## 1 Introduction and background
* overview to NLP & IR Problem
* A simple search pipeline: inverted indexing + BM25 ranking
* A full search pipeline: multi-doc query-based QA with GPT-3
* Case study: Visconda (multi-stage Retrieval + Few-shot QA with GPT-3)

## 2 From text classification to reranking
* overview of different architectures: encoder-only (BERT), decoder-only (GPT), and encoder-decoder (T5)
* Simple classification with BERT, T5, GPT-3
* BERT as reranker
* How to deal with long documents

## 3 Dense + Sparse representations
* encoding document and queries as dense vectors
* model variations (ColBERT, COIL)
* Document expansion (doc2query)
* Learned sparse representation (DeepCT, DeepImpact, UniCOIL, SPLADE)

## 4 IR in practice
* domain adaptation: InPars/Promptagator
* Multilingual and cross-lingual search
* knowledge distillation and quantization

## 5 Extracting and aggregating information from multiple documents
* multi-document QA, multi-document query-based summarization
* Neural topic modelling

## 6 Structured/Semi-structured Data
* knowledge graphs
* structure-aware language models
