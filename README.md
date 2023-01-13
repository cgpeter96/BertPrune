# BertPrune
bert裁剪层数并保存权重

## 裁切bert使用说明
支持裁切前topN层以及指定层数,e.g, 1,3,4层

**参数支持**
```python

usage: prune.py [-h] [-mp MODEL_PATH] [-pmp PRUNE_MODEL_PATH]
                [-sl SELECT_LAYERS]

optional arguments:
  -h, --help            show this help message and exit
  -mp MODEL_PATH, --model_path MODEL_PATH
  -pmp PRUNE_MODEL_PATH, --prune_model_path PRUNE_MODEL_PATH
  -sl SELECT_LAYERS, --select_layers SELECT_LAYERS
                        可以输入int or 1,2,3,4
```
**usage**
```python 
python prune.py -mp bert_uncase/ -pmp bert_uncase_prune/ -sl 1,2,3
python prune.py -mp bert_uncase/ -pmp bert_uncase_prune/ -sl 3
```

##  参考
[pytorch之对预训练的bert进行剪枝](https://www.cnblogs.com/xiximayou/p/15193655.html)