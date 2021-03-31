# DataTuner

You have just found the DataTuner. 
This repository provides tools for fine-tuning language models for a task.

* See [LICENSE.txt](LICENSE.txt) for license details.

* See [NOTICE.txt](NOTICE.txt) for details of third party code included in or downloaded by this code. 

* See [/paper/README.md](paper/README.md) for details about reproducing the results reported in the paper 
["Have Your Text and Use It Too! End-to-End Neural Data-to-Text Generation with Semantic Fidelity" by Hamza Harkous, Isabel Groves and Amir Saffari.](https://www.aclweb.org/anthology/2020.coling-main.218/)



## Installation

### Environment Creation

Assuming you have an existing `conda` setup, you can setup the environment with the following script. In order to activate the conda environment within the bash script, you need the location of the `conda.sh` file:

```bash
bash setup.sh  ~/miniconda3/etc/profile.d/conda.sh
```

You can update your existing environment:

```bash
conda env update -f=environment.yml
```

To start development, activate your environment:

```bash
conda activate finetune
```

Alternatively, you can always use the python binary with the absolute path, e.g.: `~/miniconda3/envs/finetune/bin/python`.

## Data 

For any task you want to fine-tune on, you need the data to be a json file containing a list of json objects, one per data point. For example:

```json
[
  {
    "question": "question text 1",
    "query": "query 1"
  },
  {
    "question": "question text 2",
    "query": "query 2 with [SpecialToken example]"
  }
]
```

The library assumes that you have placed your data in a single directory with three files: ``train.json``, ``validation.json``, and ``test.json``.

## Configuration 

Now that we have the data in shape, we need to create a new task configuration file that specifies how we want the data to be formatted and what fields should be considered. You can create new config files in the folder ``src/datatuner/lm/task_configs``.

A typical config file would look as follows:


```json
{
"name": "dataset_name",
"data_shape": [
        {
            "id": "<question>",
            "type": "special",
            "learn": false
        },
        {
            "id": "question",
            "type": "text",
            "learn": false
        },
        {
            "id": "<query>",
            "type": "special",
            "learn": false
        },
        {
            "id": "query",
            "type": "text",
            "learn": true,
            "metrics": [
                "match"
            ]
        }
    ],
"extra_special_tokens": ["[SpecialToken"],
"extra_fields": []
}
```

For each item in the data shape:

- ``type`` (required): ``special`` if special token, ``text`` if normal text.
- ``id`` (required): the special token ID if type is ``special``; the key for the text in the json data if type is ``text``
- ``learn`` (required): whether to allow the model to learn this part of the text. If false, the model masks that part during fine-tuning.
- ``metrics`` (optional): the list of metrics that the model should compute upon evaluation. Each metric should have a corresponding function with the same name in ``metrics.py``.
- ``converter`` (optional): the name of the converter function in ``converters.py`` to apply on that text field after reading the text from the file. 

The value of `extra_special_tokens` is a list of special tokens to be added to the vocabulary. 
Alternatively (especially if the list is too long or is generated automatically), you can create a text file with one special token per line and pass that as an argument during training via the `--special_tokens_file` argument.


The value of `extra_fields` is a list of additional fields to include from the input `json` files to output during evaluation, aside from the main fields used as inputs/outputs.

## Training 

The training script `train.py` can be used in single GPU or multi GPU settings.  

```bash
cd src/datatuner/lm

# single gpu
python train.py --model_checkpoint ~/data/openai-gpt/  --dataset_path ../../../data/my_dataset/  --task_config ./task_configs/my_task_config.json --n_epoch 3 --lr 1e-5

# multi gpu
python -m torch.distributed.launch --nproc_per_node=4 train.py --model_checkpoint ~/data/openai-gpt/  --dataset_path ../../../data/my_dataset/  --task_config ./task_configs/my_task_config.json --n_epoch 3 --lr 1e-5
```


## Evaluating the Model 

You can run the following to evaluate the model on any test set. The data format is the same as the training data. Notice that you have to currently specify the ``model_type`` parameter matching the model you're loading:

```bash
cd src/datatuner/lm

python ./evaluate.py --task_config ./task_configs/my_task_config.json --model_checkpoint runs/2020-01-01_01-01-01  --filename ../../../data/my_dataset/test.json --max_length 200 --model_type gpt --top_k 1

# or if you just want to evaluate the latest model you trained 
RUN=$(ls -t ./runs | head -1) && python ./evaluate.py --task_config ./task_configs/my_task_config.json --model_checkpoint runs/$RUN  --filename ../../../data/my_dataset/test.json --max_length 200 --model_type gpt  --top_k 1

# or if you want to use the latest intermediate checkpoint while the model is training:
RUN=$(ls -t ./runs | head -1) && CHECKPOINT=$(ls -t ./runs/$RUN/checkpoint* | head -1) && cp $CHECKPOINT runs/$RUN/pytorch_model.bin
``` 

During evaluation, the outputs that do not exactly match the expected outputs will be printed. Also,
the metrics will be printed (a dictionary with keys `<metric_name>_<field_name>`). At the end of evaluation, you will find the file with all the generated ouputs in the file `eval_results/<run_folder_name>/<task_name>_<test_file_name>_<model_type>_generated.json`.



# Interacting with the model

You can also interact with the models. The client will ask you to input the fields required, and it will generate the fields it learnt.

```bash
cd src/datatuner/lm

python ./evaluate.py --task_config ./task_configs/my_task_config.json --model_checkpoint runs/2020-01-01_01-01-01  --max_length 200 --model_type gpt  --top_k 1 --input

# or if you just want to evaluate the latest model you trained 
RUN=$(ls -t ./runs | head -1) && python ./evaluate.py --task_config ./task_configs/my_task_config.json --model_checkpoint runs/$RUN  --max_length 200 --model_type gpt  --top_k 1 --input
``` 