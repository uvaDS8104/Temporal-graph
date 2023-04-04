# Temporal-graph
project for network science from Weili Shi

First, download the raw csv files of MOOC, Reddit and Wikipedia online. Create the folder ./data. The the raw data will be preprocessed by 
```
python utils/preprocess_data.py --data wikipedia --bipartite
python utils/preprocess_data.py --data reddit --bipartite
python utils/preprocess_data.py --data mooc --bipartite
```

Then, to train the model for link prediciton
```
python train_self_supervised.py --use_memory --prefix tgn-attn -d wikipedia
python train_self_supervised.py --use_memory --prefix tgn-attn -d reddit
python train_self_supervised.py --use_memory --prefix tgn-attn -d mooc
```
To train the model for node classification

```
python train_supervised.py --use_memory --prefix tgn-attn -d wikipedia
python train__supervised.py --use_memory --prefix tgn-attn -d reddit
python train__supervised.py --use_memory --prefix tgn-attn -d mooc
```
