# **Requirements**
- Python (tested on 3.9.13)
- CUDA (tested on 11.3)
- transformers (tested on 4.35.2)
- Pytorch (tested on 1.12.0)
- numpy (tested on 1.21.5)
- tqdm (tested on 4.66.1)
- usjon
- opt-einsum (tested on 3.3.0)

# **Dataset**

The DocRED dataset can be downloaded following the instructions at [link](https://github.com/thunlp/DocRED/tree/master/data). The CDR and GDA datasets can be obtained following the instructions in edge-oriented graph. The expected structure of files is:
ATLOP
 |-- dataset
 |    |-- docred
 |    |    |-- train_annotated.json        
 |    |    |-- train_distant.json
 |    |    |-- dev.json
 |    |    |-- test.json
 |    |-- cdr
 |    |    |-- train_filter.data
 |    |    |-- dev_filter.data
 |    |    |-- test_filter.data
 |    |-- gda
 |    |    |-- train.data
 |    |    |-- dev.data
 |    |    |-- test.data
 |-- meta
 |    |-- rel2id.json
 
