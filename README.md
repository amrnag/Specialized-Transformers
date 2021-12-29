# Specialized-Transformers
This repository contains the source code for Specialized Transformers: Faster, Smaller and more Accurate NLP Models (https://openreview.net/forum?id=aUoV6qhY_e). This implementation builds on Huggingface's [Transformers](https://github.com/huggingface/transformers) library in Pytorch. Scripts from  the implementation of [BERT-of-Theseus](https://github.com/JetRunner/BERT-of-Theseus) are used to prepare files for submission to the GLUE test server.

## Dependencies
transformers 2.2.0

pytorch 1.9.1 

Tested on RTX 2080 Ti GPU with CUDA Version 11.4

## Installation
1. Install Huggingface's [Transformers](https://github.com/huggingface/transformers) library -- version 2.2.0 from source
2. Replace the transformers/data/processors/glue.py with the glue.py file provided in this repository
3. Place the run_predictions.py and finetune.py files provided in this repository inside the examples folder
4. To compare against the conventional finetuned models, download the GLUE dataset using this [script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e)

## Commands

## Results
