# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional
import fvcore.nn as fv_nn

import torch

import timm
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils
import time
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
# from torchmetrics.functional import accuracy
from torch.utils.tensorboard import SummaryWriter
import numpy as np


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, amp_autocast, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, writer=None, args = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    total_flops_in_epoch = 0  # Initialize FLOP count for the current epoch
    
    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()
    
    start_time = time.time()
    
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            
        if args.cosub:
            samples = torch.cat((samples,samples),dim=0)
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
         
        with amp_autocast():
            outputs = model(samples, if_random_cls_token_position=args.if_random_cls_token_position, if_random_token_rank=args.if_random_token_rank)
            # outputs = model(samples)
            if not args.cosub:
                loss = criterion(samples, outputs, targets)
            else:
                outputs = torch.split(outputs, outputs.shape[0]//2, dim=0)
                loss = 0.25 * criterion(outputs[0], targets) 
                loss = loss + 0.25 * criterion(outputs[1], targets) 
                loss = loss + 0.25 * criterion(outputs[0], outputs[1].detach().sigmoid())
                loss = loss + 0.25 * criterion(outputs[1], outputs[0].detach().sigmoid()) 

        if args.if_nan2num:
            with amp_autocast():
                loss = torch.nan_to_num(loss)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            if args.if_continue_inf:
                optimizer.zero_grad()
                continue
            else:
                sys.exit(1)

        optimizer.zero_grad()

        # if args.fvcore_flops:  # Check if you want to compute FLOPs
        #     flops = fv_nn.FlopCountAnalysis(model, samples)
        #     total_flops_in_epoch += flops.total()  # Add FLOPs for this batch to the total for the epoch

        # this attribute is added by timm on one optimizer (adahessian)
        if isinstance(loss_scaler, timm.utils.NativeScaler):
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward()
            if max_norm != None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
        
        # Compute gradient norm for each parameter
        grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm(2).item() ** 2  # L2 norm

        grad_norm = grad_norm ** 0.5  # Take square root to get L2 norm

        # Log the gradient norm
        metric_logger.update(grad_norm=grad_norm)

        # Compute accuracy (acc1)
        if args.is_binary:  # Binary classification
            # Get predicted class (0 or 1) by taking argmax across the second dimension (axis=1)
            _, preds = torch.max(outputs, dim=1)

            # Flatten targets to match the format of predicted class labels
            # If targets are one-hot encoded (shape [batch_size, 2]), get the index of the true class
            targets = targets.argmax(dim=1) if targets.dim() == 2 else targets
            acc1 = accuracy_score(y_true=targets.detach().cpu().numpy(), y_pred=preds.detach().cpu().numpy())
            metric_logger.update(acc1=acc1, n=samples.size(0))

        # Log the accuracy to TensorBoard
        if writer:
            writer.add_scalar('Training/Accuracy', acc1, epoch)
        
        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    end_time = time.time()
    fps_epoch = len(data_loader.dataset) / (end_time - start_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    
    # Log metrics to TensorBoard once per epoch
    if writer:
        writer.add_scalar('Loss/Train', metric_logger.loss.global_avg, epoch)
        writer.add_scalar('Training/loss', metric_logger.loss.global_avg, epoch)
        writer.add_scalar('Learning Rate', metric_logger.meters['lr'].global_avg, epoch)
        writer.add_scalar('train/learning_rate', metric_logger.meters['lr'].global_avg, epoch)
        writer.add_scalar('Training/grad_norm', metric_logger.meters['grad_norm'].global_avg, epoch)
        writer.add_scalar("Training/FPS", fps_epoch, epoch)
        writer.add_scalar("Accuracy/Train", metric_logger.meters['acc1'].global_avg, epoch)


    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# def evaluate(data_loader, model, device, amp_autocast, is_binary=False):
@torch.no_grad()
def evaluate_epoch(data_loader, model, device, amp_autocast, epoch, writer, is_binary=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    # For storing true labels and predictions to calculate metrics later
    all_labels = []
    all_preds = []
    all_probs = []  # To store predicted probabilities for AUC-ROC calculation

    # Initialize variables for tracking samples and steps
    total_samples = 0
    total_steps = 0

    # Track evaluation start time
    start_time = time.time()

    for images, target in metric_logger.log_every(data_loader, len(data_loader), header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with amp_autocast():
            output = model(images)
            loss = criterion(output, target)

        # Count the number of samples and steps processed
        total_samples += images.size(0)
        total_steps += 1

        # Update metrics (loss, accuracy, etc.)
        metric_logger.update(loss=loss.item())

        # For binary classification, calculate top-1 accuracy
        if is_binary:
            preds = output.argmax(dim=1, keepdim=True)
            acc1 = accuracy(output, target, topk=(1,))[0]
            metric_logger.meters['acc1'].update(acc1.item(), n=images.size(0))

            # Calculate precision, recall, and F1 score (weighted)
            precision = precision_score(target.cpu(), preds.cpu(), average='weighted', zero_division=0)
            recall = recall_score(target.cpu(), preds.cpu(), average='weighted', zero_division=0)
            f1 = f1_score(target.cpu(), preds.cpu(), average='weighted', zero_division=0)

            # Update the metric logger
            metric_logger.meters['precision'].update(precision, n=images.size(0))
            metric_logger.meters['recall'].update(recall, n=images.size(0))
            metric_logger.meters['f1_score'].update(f1, n=images.size(0))

            # Store true labels and predicted probabilities for AUC-ROC calculation
            all_labels.append(target.cpu())
            all_preds.append(preds.cpu())
            all_probs.append(output.softmax(dim=1).cpu())  # Softmax for probabilities

        else:  # For non-binary classification (multi-class)
            preds = output.argmax(dim=1, keepdim=True)
            acc1 = accuracy(output, target, topk=(1,))[0]
            acc5 = accuracy(output, target, topk=(5,))[0]

            # Calculate precision, recall, and F1 score (weighted) for multi-class
            precision = precision_score(target.cpu(), preds.cpu(), average='weighted', zero_division=0)
            recall = recall_score(target.cpu(), preds.cpu(), average='weighted', zero_division=0)
            f1 = f1_score(target.cpu(), preds.cpu(), average='weighted', zero_division=0)

            # Update the metric logger
            metric_logger.meters['acc1'].update(acc1.item(), n=images.size(0))
            metric_logger.meters['acc5'].update(acc5.item(), n=images.size(0))
            metric_logger.meters['precision'].update(precision, n=images.size(0))
            metric_logger.meters['recall'].update(recall, n=images.size(0))
            metric_logger.meters['f1_score'].update(f1, n=images.size(0))

            # Store true labels and predicted probabilities for AUC-ROC calculation
            all_labels.append(target.cpu())
            all_preds.append(preds.cpu())
            all_probs.append(output.softmax(dim=1).cpu())  # Softmax for probabilities

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    # Track total runtime
    total_time = time.time() - start_time
    metric_logger.meters['runtime'].update(total_time)

    # Calculate samples and steps per second
    samples_per_second = total_samples / total_time
    steps_per_second = total_steps / total_time

    # Calculate AUC-ROC (for both binary and multi-class)
    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)
    all_probs = torch.cat(all_probs)

    if is_binary:
        # AUC-ROC for binary classification (use probability of positive class)
        auc_roc = roc_auc_score(all_labels.numpy(), all_probs[:, 1].numpy())  # Assuming output has 2 classes
    else:
        # AUC-ROC for multi-class classification (use softmax probabilities)
        auc_roc = roc_auc_score(all_labels.numpy(), all_probs.numpy(), multi_class='ovr', average='weighted')

    # Update the metric logger for AUC-ROC
    metric_logger.meters['auc_roc'].update(auc_roc, n=all_labels.size(0))

    # Print out the results for this epoch
    if is_binary:
        print('* Acc@1 {top1.global_avg:.3f} Precision {precision.global_avg:.3f} '
              'Recall {recall.global_avg:.3f} F1 {f1_score.global_avg:.3f} AUC-ROC {auc_roc.global_avg:.3f} '
              'loss {losses.global_avg:.3f} runtime {runtime.global_avg:.3f}'
              .format(top1=metric_logger.acc1, precision=metric_logger.meters['precision'],
                      recall=metric_logger.meters['recall'], f1_score=metric_logger.meters['f1_score'],
                      auc_roc=metric_logger.meters['auc_roc'], losses=metric_logger.loss,
                      runtime=metric_logger.meters['runtime']))
    else:
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} Precision {precision.global_avg:.3f} '
              'Recall {recall.global_avg:.3f} F1 {f1_score.global_avg:.3f} AUC-ROC {auc_roc.global_avg:.3f} '
              'loss {losses.global_avg:.3f} runtime {runtime.global_avg:.3f}'
              .format(top1=metric_logger.acc1, top5=metric_logger.acc5, precision=metric_logger.meters['precision'],
                      recall=metric_logger.meters['recall'], f1_score=metric_logger.meters['f1_score'],
                      auc_roc=metric_logger.meters['auc_roc'], losses=metric_logger.loss,
                      runtime=metric_logger.meters['runtime']))

    # Log all metrics to TensorBoard once per evaluation (after processing entire dataset)
    writer.add_scalar(f'Evaluation/loss', metric_logger.loss.global_avg, epoch)
    writer.add_scalar(f'Evaluation/accuracy', metric_logger.acc1.global_avg / 100, epoch)
    writer.add_scalar(f'Evaluation/Accuracy', metric_logger.acc1.global_avg / 100, epoch)
    if not is_binary:
        writer.add_scalar(f'Evaluation/top5_accuracy', metric_logger.acc5.global_avg / 100, epoch)
    writer.add_scalar(f'Evaluation/precision', metric_logger.meters['precision'].global_avg, epoch)
    writer.add_scalar(f'Evaluation/Precision', metric_logger.meters['precision'].global_avg, epoch)
    writer.add_scalar(f'Evaluation/recall', metric_logger.meters['recall'].global_avg, epoch)
    writer.add_scalar(f'Evaluation/Recall', metric_logger.meters['recall'].global_avg, epoch)
    writer.add_scalar(f'Evaluation/f1_score', metric_logger.meters['f1_score'].global_avg, epoch)
    writer.add_scalar(f'Evaluation/F1 Score', metric_logger.meters['f1_score'].global_avg, epoch)
    writer.add_scalar(f'Evaluation/auc-roc', metric_logger.meters['auc_roc'].global_avg, epoch)
    writer.add_scalar(f'Evaluation/runtime', metric_logger.meters['runtime'].global_avg, epoch)
    writer.add_scalar(f'Evaluation/samples_per_second', samples_per_second, epoch)
    writer.add_scalar(f'Evaluation/steps_per_second', steps_per_second, epoch)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, amp_autocast, is_binary=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with amp_autocast():
            output = model(images)
            loss = criterion(output, target)
        
        if is_binary:
            acc = accuracy(output, target, topk=(1,))
        else:
            # For multi-class classification, we compute top-1 and top-5 accuracy
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())

        if is_binary:
            metric_logger.meters['acc1'].update(acc[0].item(), n=batch_size)  # Binary accuracy
        else:
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
            
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if is_binary:
        print('* Binary Accuracy {acc.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(acc=metric_logger.meters['acc1'], losses=metric_logger.loss))
    else:
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
