python train.py --data_dir ./dataset/docred \
--transformer_type roberta \
--model_name_or_path roberta-large \
--train_file train_annotated.json \
--dev_file dev.json \
--test_file test.json \
--distant_file distant.json \
--rel_file rel_file.json \
--train_mode 3\
--train_batch_size 16 \
--test_batch_size 16 \
--gradient_accumulation_steps 1 \
--num_labels 4 \
--learning_rate 2e-5 \
--learning_rate2 3e-6 \
--max_grad_norm 1.0 \
--warmup_ratio 0.06 \
--num_train_epochs 20.0 \
--seed 66 \
--num_class 97
