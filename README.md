# Self-Adaptive Control of Temperature
This is the code for our paper *Learning When to Concentrate or Divert Attention: Self-Adaptive Attention Temperature for Neural Machine Translation*, https://www.aclweb.org/anthology/papers/D/D18/D18-1331/

***********************************************************

## Requirements
* Ubuntu 16.0.4
* Python 3.6
* Pytorch >= 0.4
* pyrouge


**************************************************************

## Preprocessing
```
python3 preprocess.py -load_data path_to_data -save_data path_to_store_data 
```
Remember to put the data into a folder and name them *train.src*, *train.tgt*, *valid.src*, *valid.tgt*, *test.src* and *test.tgt*, and make a new folder inside called *data*

***************************************************************

## Training
```
python3 train.py -log log_name -config config_yaml -gpus id
```

****************************************************************

## Evaluation
```
python3 train.py -log log_name -config config_yaml -gpus id -restore checkpoint -mode eval
```

*******************************************************************

# Citation
If you use this code for your research, please cite the paper this code is based on: <a href="https://www.aclweb.org/anthology/papers/D/D18/D18-1331/">Learning When to Concentrate or Divert Attention: Self-Adaptive Attention Temperature for Neural Machine Translation</a>:.
```
@inproceedings{SACT,
    title = "Learning When to Concentrate or Divert Attention: Self-Adaptive Attention Temperature for Neural Machine Translation",
    author = "Lin, Junyang  and
      Sun, Xu  and
      Ren, Xuancheng  and
      Li, Muyu  and
      Su, Qi",
    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
    month = oct # "-" # nov,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D18-1331",
    pages = "2985--2990"
}
```
