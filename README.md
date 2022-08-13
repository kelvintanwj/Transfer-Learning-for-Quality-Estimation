# Transfer-Learning-for-Quality-Estimation
This project demonstrates the benefits of using Monte Carlo Dropout and upsampling in a Quality Estimation (QE) task. The training and evaluation of model is done using data from [WMT21 Shared Task: Quality Estimation](https://www.statmt.org/wmt21/quality-estimation-task.html). 

Specifically, I will focus on estimating the quality of neural machine translations by finetuning a [XLM-RoBERTa](https://huggingface.co/docs/transformers/model_doc/xlm-roberta) model using language pair sentences, each containing a source sentence and a neural machine translated sentence. 
