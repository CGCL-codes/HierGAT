# HierGAT

This is the implementation of "Entity Resolution via Hierarchical Graph Attention Network" (SIGMOD'22)

## Environment

* Python 3.7
* PyTorch 1.4
* HuggingFace Transformers
* NLTK (for 1-N ER problem)

You should run `pip install -r requirements.txt` first.

## Datasets

The raw datasets can be found at

* Magellan: https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md
* WDC: http://webdatacommons.org/largescaleproductcorpus/v2/index.html
* DI2KG: http://di2kg.inf.uniroma3.it/datasets.html

## Train HierGAT

```
python train.py \ 
	--task Amazon \
	--batch_size 32 \
	--max_len 256 \
	--lr 1e-5 \
	--n_epochs 10 \
	--finetuning \
	--split \
	--lm bert
```

- `--task`: the name of the tasks (see `task.json`)
- `--batch_size`, `--max_len`, `--lr`, `--n_epochs`: the batch size, max sequence length, learning rate, and the number of epochs
- `--split`: whether to split the attribute, should always be turned on
- `--finetuning`: whether to finetune the LM, should always be turned on
- `--lm`: the language model. We now support `bert`, `distilbert`, `xlnet`, `roberta`, `roberta-large` (`bert` by default)
  - If you want to load the model file locally, you can configure the `--lm_path`

##  Train HierGAT+

```
python train_n.py \ 
	--task N/Amazon \
	--su_len 10 \
	--finetuning \
	--split \
	--lm bert
```

Same as HierGAT, with one additional parameter:

* `--su_len`: max entity-level context sequence length

# Cite

Please cite our paper if you use this code in your own work:

```
@inproceedings{yao2022hiergat,
  title={Entity Resolution via Hierarchical Graph Attention Network},
  author={Yao, Dezhong and Gu, Yuhong and Cong, Gao and Jin, Hai and Lv, Xinqiao},
  booktitle={{SIGMOD} '22: International Conference on Management of Data, Philadelphia, PA, USA, June 12-17, 2022},
  year={2022}
}
```
