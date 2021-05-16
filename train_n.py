import argparse
import json
import os
import time

from model.ceval import eval_on_task
from model.dataset import Dataset, get_tokenizer
from model.cmodel import TranHGAT
from model.summarize import Summarizer

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter
from torch.utils import data
from transformers import AdamW, get_linear_schedule_with_warmup


def train(model, train_set, optimizer, scheduler=None, batch_size=32, su=None):
    iterator = data.DataLoader(dataset=train_set, batch_size=batch_size,
                               shuffle=False, num_workers=1, collate_fn=Dataset.pad)
    su_iterator = data.DataLoader(dataset=su, batch_size=batch_size,
                                  shuffle=False, num_workers=1, collate_fn=Dataset.padJoin)
    classifier_criterion = nn.CrossEntropyLoss()

    model.train()
    for i, (batch, su_batch) in enumerate(zip(iterator, su_iterator)):
        # for monitoring
        _, xs, y, _, masks = batch
        _, _, zs, _, _, _ = su_batch
        _y = y

        # forward
        optimizer.zero_grad()
        logits, y, _ = model(xs, zs, y, masks)

        logits = logits.view(-1, logits.shape[-1])
        y = y.view(-1)
        loss = classifier_criterion(logits, y)

        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        if i % 10 == 0:  # monitoring
            print(f"step: {i}, loss: {loss.item()}")
            del loss


def initialize_and_train(trainset, validset, testset, attr_num, args, run_tag,
                         trainset_su=None, validset_su=None, testset_su=None):
    padder = Dataset.pad
    valid_iter = data.DataLoader(dataset=validset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=0, collate_fn=padder)
    test_iter = data.DataLoader(dataset=testset, batch_size=args.batch_size,
                                shuffle=False, num_workers=0, collate_fn=padder)

    valid_su_iter = data.DataLoader(dataset=validset_su, batch_size=args.batch_size,
                                    shuffle=False, num_workers=0, collate_fn=Dataset.padJoin)
    test_su_iter = data.DataLoader(dataset=testset_su, batch_size=args.batch_size,
                                   shuffle=False, num_workers=0, collate_fn=Dataset.padJoin)

    # initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TranHGAT(attr_num, device, args.finetuning, lm=args.lm, lm_path=args.lm_path)
    if device == 'cpu':
        optimizer = AdamW(model.parameters(), lr=args.lr)
    else:
        model = model.cuda()
        optimizer = AdamW(model.parameters(), lr=args.lr)

    # learning rate scheduler
    num_steps = (len(trainset) // args.batch_size) * args.n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=num_steps)

    # create logging directory
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    writer = SummaryWriter(log_dir=args.logdir)

    # start training
    best_dev_f1 = best_test_f1 = 0.0
    epoch = 1
    while epoch <= args.n_epochs:
        start = time.time()
        train(model, trainset, optimizer, scheduler=scheduler,
              batch_size=args.batch_size, su=trainset_su)
        print("train time: ", time.time() - start)

        print(f"=========eval at epoch={epoch}=========")
        dev_f1, test_f1 = eval_on_task(epoch, model, valid_iter, test_iter, valid_su_iter, test_su_iter,
                                       writer, run_tag)

        if dev_f1 > 1e-6:
            epoch += 1
            if args.save_model:
                if dev_f1 > best_dev_f1:
                    best_dev_f1 = dev_f1
                    torch.save(model.state_dict(), run_tag + '_dev.pt')
                if test_f1 > best_test_f1:
                    best_test_f1 = dev_f1
                    torch.save(model.state_dict(), run_tag + '_test.pt')

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="N/Amazon-Google")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=17)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--su_len", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    parser.add_argument("--logdir", type=str, default="checkpoints/")
    parser.add_argument("--lm_path", type=str, default=None)
    parser.add_argument("--split", dest="split", action="store_true")
    parser.add_argument("--lm", type=str, default='bert')

    args = parser.parse_args()

    # only a single task for baseline
    task = args.task

    # create the tag of the run
    run_tag = '%s_lr=%s_id=%d' % (task, args.lr, args.run_id)
    run_tag = run_tag.replace('/', '_')

    # load task configuration
    configs = json.load(open('task.json'))
    configs = {conf['name']: conf for conf in configs}
    config = configs[task]

    trainset = config['trainset']
    validset = config['validset']
    testset = config['testset']
    category = config['category']

    # load train/dev/test sets
    train_dataset = Dataset(trainset, category, args.lm, args.lm_path, split=args.split)
    valid_dataset = Dataset(validset, category, args.lm, args.lm_path, split=args.split)
    test_dataset = Dataset(testset, category, args.lm, args.lm_path, split=args.split)

    summarizer = Summarizer(config, lm=args.lm, lm_path=args.lm_path)
    trainset_su = summarizer.transform_file(trainset, max_len=args.su_len, batch_size=args.batch_size, overwrite=True)
    validset_su = summarizer.transform_file(validset, max_len=args.su_len, batch_size=args.batch_size, overwrite=True)
    testset_su = summarizer.transform_file(testset, max_len=args.su_len, batch_size=args.batch_size, overwrite=True)

    train_dataset_su = Dataset(trainset_su, category, args.lm_path, split=args.split)
    valid_dataset_su = Dataset(validset_su, category, args.lm_path, split=args.split)
    test_dataset_su = Dataset(testset_su, category, args.lm_path, split=args.split)
    initialize_and_train(train_dataset, valid_dataset, test_dataset,
                            train_dataset.get_attr_num(), args, run_tag,
                            train_dataset_su, valid_dataset_su, test_dataset_su)
