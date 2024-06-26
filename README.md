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
- ATLOP
  - dataset
    - docred
      - train_annotated.json
      - train_distant.json
      - dev.json
      - test.json
      - rel_file.json
    - cdr
      - train_filter.data
      - dev_filter.data
      - test_filter.data
      - -rel_file.json
    - gda
      - train.data
      - dev.data
      - test.data
      - rel_file.json
    - meta
      - rel2id.json

# **Training and Evaluation**

Train the teacher model on distant dataset:

```bash
sh scripts/train_teacher_roberta.sh # for RoBERTa
```
finetune teacher model on human-annotated dataset:

```bash
sh scripts/finetune_teacher_roberta.sh # for RoBERTa
```
infer teacher model on distant dataset:

```bash
sh scripts/infer_teacher_roberta.sh # for RoBERTa
```
distillation the student model on distant dataset:

```bash
sh scripts/distill_student_roberta.sh # for RoBERTa
```
finetune the student model on distant dataset:

```bash
sh scripts/finetune_student_roberta.sh # for RoBERTa
```

The program will generate a test file **result.zip** in the official evaluation format. You can compress and submit it to Colab for the official test score.
