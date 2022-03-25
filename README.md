# MTG-Jamendo FCN baseline

PyTorch Lightning implementation of the baseline FCN auto-tagging model.

Requires Python 3.9+

* Clone the mtg-jamendo-dataset (e.g. path/to/mtg-jamendo-dataset)
* Download and extract MTG-Jamendo spectrograms (e.g. path/to/specs)


If you want to use W&B:
* `pip install wandb`
* set the environment variables: `WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_PROJECT`


```shell
python -m src.train path/to/specs path/to/mtg-jamendo-dataset --max_epochs 3 --num-workers 12 --gpus 1 --batch-size 64
```

## Dev
Update all the revs to the latest versions

```shell
pip install pre-commit
pre-commit install
```
