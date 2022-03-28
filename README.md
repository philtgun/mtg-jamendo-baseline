# MTG-Jamendo FCN baseline

PyTorch Lightning implementation of the baseline FCN auto-tagging model.

Requires Python 3.9+

## Getting the data
* Clone the [mtg-jamendo-dataset](https://github.com/MTG/mtg-jamendo-dataset) (e.g. `path/to/mtg-jamendo-dataset`)
* Download and extract MTG-Jamendo spectrograms (e.g. `path/to/specs`)


## Running
* Create the virtualenv and install dependencies
```shell
python3.9 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

* To use W&B:
  * `pip install wandb`
  * set the environment variables: `WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_PROJECT`
(you can add those to `venv/bin/activate` script, e.g. for fish: `set -gx WANDB_API_KEY "xxxx"`)


* Training
```shell
python -m src.train path/to/specs path/to/mtg-jamendo-dataset --max_epochs 100 --num-workers 12 --gpus 1 --batch-size 32 --sampling random --output-path out/results.csv --models-path out/models
```
Or you can use the script to train each subset and split (adjust parameters inside as necessary)
```shell
./run_all.sh path/to/specs path/to/mtg-jamendo-dataset
```

## Dev

```shell
pip install pre-commit
pre-commit install
```
