import torch
import numpy as np

def calculate_metric(confusion_matrix, metrics):
    ret_metric = {}
    TP = confusion_matrix['TP']
    FP = confusion_matrix['FP']
    FN = confusion_matrix['FN']
    
    for metric in metrics:
        if metric == "mIoU":
            ret_metric[metric] = TP / (TP + FP + FN)
            ret_metric['total_mIoU'] = ret_metric[metric].mean()
        if metric == "precision":
            ret_metric[metric] = TP / (TP + FP)
            ret_metric['total_precision'] = ret_metric[metric].mean()
        if metric == "recall":
            ret_metric[metric] = TP / (TP + FN)
            ret_metric['total_recall'] = ret_metric[metric].mean()
               
        
    
    return ret_metric
         
    


def evalutate(pre_eval_results, metrics=['mIoU', 'precision', 'recall']):

    # convert list of tuples to tuple of lists, e.g.
    # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
    # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
    num_classes = pre_eval_results[0][0].shape[0]
    
    total_TP = torch.zeros(num_classes, dtype=pre_eval_results[0][0].dtype)
    total_FP = torch.zeros(num_classes, dtype=pre_eval_results[0][0].dtype)
    total_FN = torch.zeros(num_classes, dtype=pre_eval_results[0][0].dtype)
    
    confusion_matrix = {}
    
    for i in range(len(pre_eval_results)):
        intersection, union, pred_area, label_area = \
            pre_eval_results[i]
        
        TP = intersection
        FP = pred_area - intersection
        FN = label_area - intersection
        
        total_TP += TP
        total_FP += FP
        total_FN += FN
    
    
    confusion_matrix['TP'] = total_TP
    confusion_matrix['FP'] = total_FP
    confusion_matrix['FN'] = total_FN   
    
    ret_metric = calculate_metric(confusion_matrix, metrics)
    
    return ret_metric
            

