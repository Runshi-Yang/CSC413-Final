import logging
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

def valids(model, valid_loader, device):
    with torch.no_grad():
        criterion = nn.CrossEntropyLoss()
        model.eval()
        y_true = []
        y_pred = []
        y_score = []
        loss = []
        iteration = 0
        for data_sample in valid_loader:
            y = data_sample['label'].to(device)
            outputs = model(
                data_sample['ligase_pocket'].to(device),
                data_sample['target_pocket'].to(device),
                data_sample['PROTAC'].to(device),
            )
            loss_val = criterion(outputs, y)
            loss.append(loss_val.item())
            y_score = y_score + torch.nn.functional.softmax(outputs,1)[:,1].cpu().tolist()
            y_pred = y_pred + torch.max(outputs,1)[1].cpu().tolist()
            y_true = y_true + y.cpu().tolist()
            iteration += 1
        model.train()
    return loss, accuracy_score(y_true, y_pred), roc_auc_score(y_true, y_score)


def train(model, lr=0.0001, epoch=30, weight_decay=0, train_loader=None, valid_loader=None, device=None, writer=None, LOSS_NAME=None, batch_size = None):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    _ = valids(model, valid_loader, device)
    best_val_loss = 1e6
    train_losses = []
    val_losses = []

    mean_train_losses = []
    mean_val_losses = []

    early_stopping_counter = 0
    criterion = nn.CrossEntropyLoss()
    for epo in range(epoch):
        total_num = 0
        train_loss = []
        for data_sample in train_loader:
            outputs = model(
                data_sample['ligase_pocket'].to(device),
                data_sample['target_pocket'].to(device),
                data_sample['PROTAC'].to(device),
            )
            total_num += batch_size
            y = data_sample['label'].to(device)
            loss = criterion(outputs, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss.append(loss.item())
        val_loss, val_acc, auroc = valids(model, valid_loader, device)

        mean_train_loss = np.mean(train_loss)
        mean_val_loss = np.mean(val_loss)

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter > 20:
            logging.info("Validation loss has not improved in 20 epochs, stopping early")
            logging.info("Obtained lowest validation loss of: {}".format(best_val_loss))
            torch.save(model, f"model/{LOSS_NAME}.pt")
            mean_train_losses.append(mean_train_loss)
            mean_val_losses.append(mean_val_loss)
            return model, mean_train_losses, mean_val_losses, val_acc

        logging.info('Train epoch %d, loss: %.4f, val_loss: %.4f, val_acc: %.4f, val_auroc: %.4f' % (epo, mean_train_loss, mean_val_loss, val_acc, auroc))
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        mean_train_losses.append(mean_train_loss)
        mean_val_losses.append(mean_val_loss)

    torch.save(model, f"model/{LOSS_NAME}.pt")
    return model, mean_train_losses, mean_val_losses, val_acc

def test(model, test_loader, device):
    with torch.no_grad():
        criterion = nn.CrossEntropyLoss()
        model.eval()
        y_true = []
        y_pred = []
        y_score = []
        loss = []
        iteration = 0
        for data_sample in test_loader:
            y = data_sample['label'].to(device)
            outputs = model(
                data_sample['ligase_pocket'].to(device),
                data_sample['target_pocket'].to(device),
                data_sample['PROTAC'].to(device),
            )
            loss_val = criterion(outputs, y)
            loss.append(loss_val.item())
            y_score = y_score + torch.nn.functional.softmax(outputs,1)[:,1].cpu().tolist()
            y_pred = y_pred + torch.max(outputs,1)[1].cpu().tolist()
            y_true = y_true + y.cpu().tolist()
            iteration += 1
        model.train()
    return sum(loss)/iteration, accuracy_score(y_true, y_pred), roc_auc_score(y_true, y_score)