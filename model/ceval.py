import torch
import torch.nn as nn
import sklearn.metrics as metrics

import time


def eval_classifier(model, iterator, su_iterator):
    model.eval()

    Y = []
    Y_hat = []
    loss_list = []
    total_size = 0
    with torch.no_grad():
        for i, (batch, su_batch) in enumerate(zip(iterator, su_iterator)):
            _, x, y, _, masks = batch
            _, _, z, _, _, _ = su_batch
            logits, y1, y_hat = model(x, z, y, masks)

            logits = logits.view(-1, logits.shape[-1])
            y1 = y1.view(-1)
            loss = nn.CrossEntropyLoss()(logits, y1)

            loss_list.append(loss.item() * y.shape[0])
            total_size += y.shape[0]

            Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    loss = sum(loss_list) / total_size
    print("======================================")

    accuracy = metrics.accuracy_score(Y, Y_hat)
    precision = metrics.precision_score(Y, Y_hat)
    recall = metrics.recall_score(Y, Y_hat)
    f1 = metrics.f1_score(Y, Y_hat)
    print("accuracy=%.4f" % accuracy)
    print("precision=%.4f" % precision)
    print("recall=%.4f" % recall)
    print("f1=%.4f" % f1)
    print("======================================")

    return accuracy, precision, recall, f1, loss


def eval_on_task(epoch, model, valid_iter, test_iter, valid_su_iter, test_su_iter,
                 writer, run_tag):
    print('Validation:')
    start = time.time()
    v_output = eval_classifier(model, valid_iter, valid_su_iter)
    print("valid time: ", time.time() - start)

    print('Test:')
    t_output = eval_classifier(model, test_iter, test_su_iter)

    acc, prec, recall, f1, v_loss = v_output
    t_acc, t_prec, t_recall, t_f1, t_loss = t_output
    scalars = {'acc': acc,
               'precision': prec,
               'recall': recall,
               'f1': f1,
               'v_loss': v_loss,
               't_acc': t_acc,
               't_precision': t_prec,
               't_recall': t_recall,
               't_f1': t_f1,
               't_loss': t_loss}

    # logging
    writer.add_scalars(run_tag, scalars, epoch)
    return f1, t_f1
