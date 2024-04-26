import argparse
import os

import numpy as np
import torch
from copy import deepcopy
#from apex import amp
import ujson as json
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from model import DocREModel
from utils import set_seed, collate_fn
from prepro import read_docred
from evaluation import to_official, official_evaluate
# from LargeJsonDatesets import LargeTrainDataset
import random
# import wandb, gc
import gc
from tqdm import tqdm

def train(args, model, train_features, dev_features, test_features):
    def finetune(features, optimizer, num_epoch, num_steps):
        best_score = -1                
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
        train_iterator = range(int(num_epoch))
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        print("Total steps: {}".format(total_steps))
        print("Warmup steps: {}".format(warmup_steps))
        for epoch in train_iterator:
            optimizer.zero_grad()
            for step, batch in enumerate(tqdm(train_dataloader) ):
                model.train()
                inputs = {'input_ids': batch[0].to(args.device),
                          'attention_mask': batch[1].to(args.device),
                          'labels': batch[2],
                          'entity_pos': batch[3],
                          'hts': batch[4],
                        #   'teacher_logits':batch[5]                                                   
                          }
                with torch.cuda.amp.autocast(enabled=False):
                    outputs = model(**inputs)           
                    loss = outputs / args.gradient_accumulation_steps
                

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if step % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

                    num_steps += 1

                
                if (step + 1) == len(train_dataloader) - 1 or (args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):
                    dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
#                     wandb.log(dev_output, step=num_steps)
                    print(dev_output)
                    print("loss", epoch , loss.item())
                    if dev_score > best_score:
                        best_score = dev_score
                        pred, logits = report(args, model, test_features)
                        with open("result.json", "w") as fh:
                            json.dump(pred, fh)
                        if args.save_path != "":
                            torch.save(model.state_dict(), args.save_path)
        return num_steps

    new_layer = ["extractor", "bilinear", "encoder_layer_2", "encoder_layer_1"]
    
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": args.learning_rate2},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    
    num_steps = 0
    set_seed(args)
    model.zero_grad()
      

    finetune(train_features, optimizer, num_epochss, num_steps)


def evaluate(args, model, features, tag="dev"):

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    total_loss = 0
    print("dev.....")
    for i ,batch in enumerate( tqdm(dataloader)):
        model.eval()
        gc.collect()
        torch.cuda.empty_cache()
        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                #   'labels' : batch[2],
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  
                  }

        with torch.no_grad():
            output, *_ =  model(**inputs)
            pred = output.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)
            total_loss += output[0].item()
    

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    ans = to_official(preds, features)
    if len(ans) > 0:
        best_f1, _, best_f1_ign, _ = official_evaluate(ans, args.data_dir)
    output = {
        tag + "_F1": best_f1 * 100,
        tag + "_F1_ign": best_f1_ign * 100,
        
    }
    print("dev finish......")
    return best_f1, output

def add_logits_to_features(features, logits):
    new_features = []
    for i, old_feature in enumerate(features):
        new_feature = deepcopy(old_feature)
        new_feature['teacher_logits'] = logits[i]
        # print(logits[i].shape[0], len(new_feature['hts']))
        assert logits[i].shape[0] == len(new_feature['hts'])
        new_features.append(new_feature)

    return new_features

def report(args, model, features):

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    logits = []
    print("test.....")
    # for batch in dataloader:
    print(len(dataloader))
    for i ,batch in enumerate( tqdm(dataloader)):
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  
                  }

        with torch.no_grad():
            pred, logit = model(**inputs)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)
            logits.append(logit.detach().cpu())
    
    print("test finish....")
    preds = np.concatenate(preds, axis=0).astype(np.float32)
    preds = to_official(preds, features)
    print("finish......")
    return preds, logits
    # return logits


def main():
    parser = argparse.ArgumentParser()

    #get parameter
    parser.add_argument("--data_dir", default="./dataset/docred", type=str)
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)

    parser.add_argument("--train_file", default="train_annotated.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)
    parser.add_argument("--distant_file", default="train_distant.json", type=str)
    parser.add_argument("--rel_file", default="rel_file.json", type=str)

    parser.add_argument("--save_path", default="./model_save/docred/model-teacher.pkl", type=str)
    parser.add_argument("--load_path", default="./model_save/docred/model-teacher.pkl", type=str)
    parser.add_argument("--infer_path", default="./model_save/docred/infer_log.pt", type=str)
    parser.add_argument("--train_mode", default="1", type=int)

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=8, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_labels", default=4, type=int,
                        help="Max number of labels in prediction.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--learning_rate2", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--evaluation_steps", default=-1, type=int,
                        help="Number of training steps between evaluations.")
    parser.add_argument("--seed", type=int, default=66,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=97,
                        help="Number of relation types in dataset.")

    args = parser.parse_args()
#     wandb.init(project="DocRED")

    #set device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    #set bert parameter
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )

    read = read_docred
    
    #read file
    train_file = os.path.join(args.data_dir, args.train_file)
    dev_file = os.path.join(args.data_dir, args.dev_file)
    test_file = os.path.join(args.data_dir, args.test_file)
    distant_file = os.path.join(args.data_dir, args.distant_file)
    rel_file = os.path.join(args.data_dir, args.rel_file)
    
    train_features = read(train_file, tokenizer, max_seq_length=args.max_seq_length)
    dev_features = read(dev_file, tokenizer, max_seq_length=args.max_seq_length)
    test_features = read(test_file, tokenizer, max_seq_length=args.max_seq_length)
    distant_features = read(distant_file, tokenizer, max_seq_length=args.max_seq_length)
    rel_features = read(rel_file, tokenizer, max_seq_length=args.max_seq_length)
    
    
    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    ).to(1)

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type

    set_seed(args)
    
    #get model
    model = DocREModel(config, model, rel_features, num_labels=args.num_labels)
    model.to(1)
    flag = False
    
    if args.train_mode == 1:  # Training
    #train teacher model in distant dataset        
        train(args, model, train_features, dev_features, test_features)

    elif args.train_mode == 2:     
    #finetinue teacher model in human dataset
        #load teacher model
        model.load_state_dict(torch.load(args.load_path))
        
        #finetune teacher model
        train(args, model, train_features, dev_features, test_features)
    
    elif args.train_mode == 3:
    #teacher model infernce in distant dataset
        #load teacher model
        model.load_state_dict(torch.load(args.load_path))
        #soft label
        re, logits = report(args, model, distant_features)
        #save soft label
        train_features = add_logits_to_features(distant_features, logits)
        torch.save(train_features, args.infer_path)
    
    elif args.train_mode == 4:
    #distill student model in distant distant 
        #load soft label 
        distill_features = torch.load(args.infer_path)
        #distill
        train(args, model, distill_features, dev_features, test_features)

    else:
    #fintune student model in human-annotated dataset
        #load student model
        model.load_state_dict(torch.load(args.load_path))
        #finetune student model
        train(args, model, train_features, dev_features, test_features)
    
    
if __name__ == "__main__":
    main()
