# Specialized Transformers: Faster, Smaller and more Accurate NLP Models
This repository contains the source code for Specialized Transformers: Faster, Smaller and more Accurate NLP Models (https://openreview.net/forum?id=aUoV6qhY_e). This implementation builds on Huggingface's [Transformers](https://github.com/huggingface/transformers) library in Pytorch. Scripts from  the implementation of [BERT-of-Theseus](https://github.com/JetRunner/BERT-of-Theseus) are used to prepare files for submission to the GLUE test server.

## Dependencies
transformers 2.2.0

pytorch 1.9.1 

Tested on RTX 2080 Ti GPU with CUDA Version 11.4

## Installation
1. Install Huggingface's [Transformers](https://github.com/huggingface/transformers) library (version 2.2.0) from source.
2. Replace the transformers/data/processors/glue.py with the glue.py file provided in this repository. Also replace the transformers/modeling_bert.py file with the modeling_bert.py file provided in this repository.
3. Place the run_predictions.py and finetune.py files provided in this repository inside the examples folder.
4. To compare against the conventional finetuned models, download the GLUE dataset using this [script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e).

## Commands

+ In order to ensure reproducibility, this repository also includes a train-validation split configuration for CoLA, RTE and WNLI (the three tasks where the conventional fine-tuned models achieve lowest accuracies) under `data/`. To obtain the specialized model, use `python run_predictions.py   --model_type bert   --model_name_or_path bert-base-cased   --task_name $TASK_NAME --do_train   --do_eval   --do_lower_case   --data_dir /data/$TASK_NAME/   --max_seq_length 128   --per_gpu_train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 3.0   --output_dir /tmp/$TASK_NAME --overwrite_output_dir`
+ To obtain the conventional finetuned model, use `python finetune.py   --model_type bert   --model_name_or_path bert-base-cased   --task_name $TASK_NAME --do_train   --do_eval   --do_lower_case   --data_dir $GLUE_DIR/$TASK_NAME/   --max_seq_length 128   --per_gpu_train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 3.0   --output_dir /tmp/$TASK_NAME --overwrite_output_dir`. Here, $GLUE_DIR refers to the path to the GLUE dataset downloaded using this [script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e). The entire training set is used to finetune the model (no train-validation split).
+ To analyze variance of the accuracy of the specialized model across different seeds, use `python finetune.py   --model_type bert   --model_name_or_path bert-base-cased   --task_name $TASK_NAME --do_train   --do_eval   --do_lower_case   --data_dir $GLUE_DIR/$TASK_NAME/   --max_seq_length 128   --per_gpu_train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 3.0   --output_dir /tmp/$TASK_NAME --overwrite_output_dir --from_specialized --seed $SEED`. Here, $GLUE_DIR refers to the path to the GLUE dataset downloaded using this [script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e). The entire training set is used to finetune the model (no train-validation split).

## Results
+ Results on the GLUE test set (obtained by submitting to the GLUE [server](https://gluebenchmark.com/))
<table>
    <tr align="center">
        <th>Dataset</th>
        <th>Conventional fine-tuned model</th>
        <th>Specialized model</th>
    </tr>
    <tr align="center">
        <td>CoLA</td>
        <td>52.1</td>
        <td>53.6</td>
    </tr>
    <tr align="center">
        <td>RTE</td>
        <td>61.7</td>
        <td>63.6</td>
    </tr>
      <tr align="center">
        <td>WNLI</td>
        <td>40.4</td>
        <td>65.8</td>
    </tr>
</table>

+ Results on the GLUE development set
<table>
    <tr align="center">
        <th>Dataset</th>
        <th>Conventional fine-tuned model</th>
        <th>Specialized model</th>
    </tr>
    <tr align="center">
        <td>CoLA</td>
        <td>55.23</td>
        <td>56.95</td>
    </tr>
    <tr align="center">
        <td>RTE</td>
        <td>61.73</td>
        <td>63.54</td>
    </tr>
      <tr align="center">
        <td>WNLI</td>
        <td>30.99</td>
        <td>56.34</td>
    </tr>
</table>

## Citing this work
```
@inproceedings{
anonymous2022specialized,
title={Specialized Transformers: Faster, Smaller and more Accurate {NLP} Models},
author={Anonymous},
booktitle={Submitted to The Tenth International Conference on Learning Representations },
year={2022},
url={https://openreview.net/forum?id=aUoV6qhY_e},
note={under review}
}
```
