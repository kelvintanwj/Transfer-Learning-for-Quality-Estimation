# Overview
This project demonstrates the benefits of using Monte Carlo Dropout and upsampling in a Quality Estimation (QE) task. The training and evaluation of model is done using data from [WMT21 Shared Task: Quality Estimation](https://www.statmt.org/wmt21/quality-estimation-task.html). 

Specifically, I focused on estimating the quality of neural machine translations by finetuning a [XLM-RoBERTa](https://huggingface.co/docs/transformers/model_doc/xlm-roberta) model using language pair sentences, each containing a source sentence and a neural machine translated sentence. The datasets were collected by translating sentences sampled from source language articles using state-of-the-art Transformer NMT models and annotated with a variant of Direct Assessment (DA) scores by professional translators.

Whilst the above model was able to outperform the baseline models in all language pairs (which includes 12 different language pairs) and in zero shot situations, it was unable to close the gap with the current state-of-the-art performance. 

# To use:
I recommend using GCP for the training and evaluation of model. To do so, simply import the repository into Google Cloud Platform and run the codes in en_de.ipynb.
