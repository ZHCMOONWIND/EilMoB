import os
import torch
from torch.utils import data
import argparse
import random
import numpy as np
import time
import torch.nn.functional as F
from sklearn import metrics
from models.text_models import get_text_configuration, get_text_model, initialize_tokenizer
from models.vision_models import initialize_transforms, get_vision_model, get_vision_configuration
from models.vision_text import MultiModalDataset, get_multimodal_model, get_multimodal_configuration
from utility.text_sentiment import get_text_sentiment, get_tokens_len
import torchtext.vocab as vocab
from tqdm import tqdm
import csv
import pandas as pd
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def binary_accuracy(preds, y):
    acc = metrics.accuracy_score(y, preds)
    return acc


def get_sentiment(data_names, sentiment_labels):
    sentiment_res = []
    for file_name in data_names:
        assert file_name in sentiment_labels.keys()
        sentiment_res.append(sentiment_labels[file_name])
    return sentiment_res
    
def voting_decision(logits_list):
    predictions = [torch.argmax(logit, dim=1) for logit in logits_list]
    stacked_predictions = torch.stack(predictions, dim=1)
    final_predictions = torch.mode(stacked_predictions, dim=1).values
    return final_predictions

def train(args, text_tools, vision_model, text_model, multimodal_model, loader, optimizer, criterion, scheduler, device, vision_optimizer, vision_scheduler, text_optimizer, text_scheduler, mode):
    epoch_loss = 0
    epoch_acc = 0
    vision_model.train()
    text_model.eval()
    multimodal_model.train()
    if mode == 'align':
        progress_bar = tqdm(enumerate(loader), total=len(loader), desc="Aligning", leave=False)
    else:
        progress_bar = tqdm(enumerate(loader), total=len(loader), desc="Training", leave=False)
    for iter, data in progress_bar:
        data_names, vision_data, text_data, llm_data, label = data[0], data[1], data[2], data[3], data[4]
               
        test_lens = get_tokens_len(text_data, device)
        llm_lens = get_tokens_len(llm_data, device)
        text_sentiments = get_text_sentiment(text_data, device)
        llm_sentiments = get_text_sentiment(llm_data, device)
        text_ids = text_tools['tokenizer'](list(text_data), padding='longest', truncation=True, return_tensors='pt')['input_ids'].to(device)
        llm_ids = text_tools['tokenizer'](list(llm_data), padding='longest', truncation=True, return_tensors='pt')['input_ids'].to(device)
        vision_data, text_ids, llm_ids, label = vision_data.to(device), text_ids.to(device), llm_ids.to(device), label.to(device)

        vision_embeddings = vision_model(vision_data)
        text_embeddings = text_model(text_ids)
        llm_embeddings = text_model(llm_ids)
        logits, loss = multimodal_model(vision_embeddings, text_embeddings, llm_embeddings, test_lens, llm_lens, text_sentiments, llm_sentiments, mode, label)
        predictions = voting_decision(logits)
        correct = (predictions == label).sum().item()
        acc = correct / predictions.size(0)
        vision_optimizer.zero_grad() 
        text_optimizer.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        vision_optimizer.step()
        text_optimizer.step()
        optimizer.step()
        vision_scheduler.step()
        text_scheduler.step()
        scheduler.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc
        progress_bar.set_postfix(loss=epoch_loss / (iter + 1) ,acc=epoch_acc / (iter + 1))
    return epoch_loss / len(loader), epoch_acc / len(loader)


def evaluate(args, text_tools, epoch, all_epoch, best_acc, vision_model, text_model, multimodal_model, loader, criterion, device, mode):
    epoch_loss = 0
    vision_model.eval()
    text_model.eval()
    multimodal_model.eval()
    
    preds = []
    labels = []
    evaluate_flag = []
    progress_bar = tqdm(enumerate(loader), total=len(loader), desc="Evaluating", leave=False)
    with torch.no_grad():
        for iter, data in progress_bar:
            data_names, vision_data, text_data, llm_data, label = data[0], data[1], data[2], data[3], data[4]

            text_lens = get_tokens_len(text_data, device)
            llm_lens = get_tokens_len(llm_data, device)
            text_sentiments = get_text_sentiment(text_data, device)
            llm_sentiments = get_text_sentiment(llm_data, device)

            text_ids = text_tools['tokenizer'](list(text_data), padding='longest', truncation=True, return_tensors='pt')['input_ids'].to(device)
            llm_ids = text_tools['tokenizer'](list(llm_data), padding='longest', truncation=True, return_tensors='pt')['input_ids'].to(device)
            vision_data, text_ids, llm_ids, label = vision_data.to(device), text_ids.to(device), llm_ids.to(device), label.to(device)

            vision_embeddings = vision_model(vision_data)
            text_embeddings = text_model(text_ids)
            llm_embeddings = text_model(llm_ids)

            logits, loss = multimodal_model(vision_embeddings, text_embeddings, llm_embeddings, text_lens, llm_lens, text_sentiments, llm_sentiments, mode, label)
            
            predictions = voting_decision(logits)
            preds.extend(predictions.cpu().detach().numpy().tolist())
            labels.extend(label.cpu().detach().numpy().tolist())
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=epoch_loss / (iter + 1))
    acc = metrics.accuracy_score(labels, preds)
    binary_f1 = metrics.f1_score(labels[:], preds[:])
    binary_precision = metrics.precision_score(labels[:], preds[:])
    binary_recall = metrics.recall_score(labels[:], preds[:])
    macro_f1 = metrics.f1_score(labels[:], preds[:], average='macro')
    macro_precision = metrics.precision_score(labels[:], preds[:], average='macro')
    macro_recall = metrics.recall_score(labels[:], preds[:], average='macro')
    best_acc = max(best_acc, acc)
    print('Epoch: {}/{}:  Macro F1:  {} Macro Precision: {}  Macro Recall: {}  Binary F1: {}  Binary Precision: {}  Binary Recall: {}  Acc: {}  Best Acc: {}'.format(
        epoch, all_epoch, macro_f1, macro_precision, macro_recall, binary_f1, binary_precision, binary_recall, acc, best_acc
    ))
    evaluate_flag.append(macro_f1)
    evaluate_flag.append(macro_precision)
    evaluate_flag.append(macro_recall)
    evaluate_flag.append(binary_f1)
    evaluate_flag.append(binary_precision)
    evaluate_flag.append(binary_recall)
    return epoch_loss / len(loader), acc, evaluate_flag


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def run(args):
    set_seed(args.seed)
    device = torch.device('cuda', args.gpu)
    tokenizer = initialize_tokenizer(args)
    text_tools = { 'tokenizer': tokenizer }
    vision_transforms = initialize_transforms(args)
    train_set = MultiModalDataset(text_tools, vision_transforms, args, 'train')
    test_set = MultiModalDataset(text_tools, vision_transforms, args, 'test')
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    vision_model = get_vision_model(args)
    text_model = get_text_model(args, embedding=True)
    multimodal_model = get_multimodal_model(args)

    if not args.train_new_or_old:
        checkpoint = torch.load('./saved_models/test.pth')
        print(checkpoint['acc'])
        vision_model.load_state_dict(checkpoint['vision_model'])
        text_model.load_state_dict(checkpoint['text_model'])
        multimodal_model.load_state_dict(checkpoint['multimodal'])

    vision_optimizer, vision_scheduler, _ = get_vision_configuration(args, vision_model)
    text_optimizer, text_scheduler, _ = get_text_configuration(args, text_model)
    optimizer, scheduler, criterion = get_multimodal_configuration(args, multimodal_model)
    
    vision_model.to(device)
    text_model.to(device)
    multimodal_model.to(device)
    criterion.to(device)
    best_test_acc = -float('inf')
    ans = []
    for epoch in range(1, args.epoch+1):
        start_time = time.time()
        acc_list = []
        train_loss, train_acc = train(args, text_tools, vision_model, text_model, multimodal_model, train_loader, optimizer, criterion, scheduler, device, vision_optimizer, vision_scheduler, text_optimizer, text_scheduler)

        
        test_loss, test_acc, evaluate_flag= evaluate(args, text_tools, epoch, args.epoch, best_test_acc, vision_model, text_model, multimodal_model, test_loader, criterion, device)
        acc_list.append(train_acc)
        acc_list.append(train_loss)
        acc_list.append(test_acc)
        acc_list.append(test_loss)
        for i in evaluate_flag:
            acc_list.append(i)
        ans.append(acc_list)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save({
                'vision_model': vision_model.state_dict(),
                'text_model': text_model.state_dict(),
                'multimodal': multimodal_model.state_dict(),
                'acc': test_acc
            }, os.path.join(args.save_dir, args.save_name))
        else :
            torch.save({
                'vision_model': vision_model.state_dict(),
                'text_model': text_model.state_dict(),
                'multimodal': multimodal_model.state_dict(),
                'acc': test_acc
            }, os.path.join(args.save_dir, 'test_new.pth'))
        print(f'Epoch: {epoch:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\tTest. Loss: {test_loss:.3f} | Test. Acc: {test_acc*100:.2f}%')
        pd.DataFrame(ans,columns=['Train Acc','Train Loss','Test Acc','Test Loss','macro_f1', 'macro_precision', 'macro_recall', 'binary_f1', 'binary_precision', 'binary_recall']).to_csv('./acc.csv',index = False)


def main():
    parser = argparse.ArgumentParser(description='')

    # save information
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('--save_name', type=str, default='test.pth')
    parser.add_argument('--seed', default=12345, type=int, help='seed for initializing training.')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    
    # train information
    parser.add_argument('--vision_backbone', type=str, default='vit')
    parser.add_argument('--vision_model', type=str, default='base')
    parser.add_argument('--text_backbone', type=str, default='bert')
    parser.add_argument('--text_model', type=str, default='base')
    parser.add_argument('--epoch', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--vision_lr', type=float, default=2e-5)
    parser.add_argument('--vision_weight_decay', type=float, default=1e-5)
    parser.add_argument('--text_lr', type=float, default=2e-5)
    parser.add_argument('--text_weight_decay', type=float, default=1e-5)
    parser.add_argument('--multimodal_lr', type=float, default=2e-5)
    parser.add_argument('--multimodal_weight_decay', type=float, default=1e-5)

    # dataset configuration
    parser.add_argument('--train_text_path', type=str, default='../text_data/train.txt')
    parser.add_argument('--val_text_path', type=str, default='../text_data/val.txt')
    parser.add_argument('--test_text_path', type=str, default='../text_data/test.txt')
    parser.add_argument('--train_image_path', type=str, default='../imgs/train')
    parser.add_argument('--val_image_path', type=str, default='../imgs/val')
    parser.add_argument('--test_image_path', type=str, default='../imgs/test')

    parser.add_argument('--train_llm_path', type=str, default='../llm/train.txt')
    parser.add_argument('--val_llm_path', type=str, default='../llm/val.txt')
    parser.add_argument('--test_llm_path', type=str, default='../llm/test.txt')

    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--train_new_or_old',default=True)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()

