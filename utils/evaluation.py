#https://github.com/sirius8050/Expected-Calibration-Error/blob/master/ECE.model_output


import numpy as np
import torch
from torchmetrics import AUROC #https://torchmetrics.readthedocs.io/en/stable/classification/auroc.html
from torchmetrics.classification import MulticlassCalibrationError
from torchmetrics.classification import MulticlassSpecificity
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score


def ece_score(model_output, model_labels, num_classes, n_bins=10):
    metric = MulticlassCalibrationError(num_classes=num_classes, n_bins=n_bins, norm='l1')
    result = metric(model_output, model_labels)
    result = np.array(result)
    result = float(result)
    return result

#     # Output and labels to numpy arrays
#     model_output = np.array(model_output)
#     model_labels = np.array(model_labels)

#     # If needed modify labels
#     if model_labels.ndim > 1:
#         model_labels = np.argmax(model_labels, axis=1)

#     # Get prediction of each run (always the maxium value)
#     model_output_index = np.argmax(model_output, axis=1)

#     # Get label index to each prediction
#     model_output_value = []
#     for i in range(model_output.shape[0]):
#         model_output_value.append(model_output[i, model_output_index[i]])
#     model_output_value = np.array(model_output_value)

#     # Create bins for accuracy and confidence
#     acc, conf = np.zeros(n_bins), np.zeros(n_bins)
#     Bm = np.zeros(n_bins)

#     # For each bin
#     for m in range(n_bins):
#         a, b = m / n_bins, (m + 1) / n_bins  # First iteration: 0.1 and 0.2

#         # for each model output
#         for i in range(model_output.shape[0]):
#             if model_output_value[i] > a and model_output_value[i] <= b:  # If output value (confidence) between those numbers, fill bin
#                 Bm[m] += 1  # Fill overall bin
#                 conf[m] += model_output_value[i] # Fill confidence bin
#                 if model_output_index[i] == model_labels[i]:  # If its also correctly classified
#                     acc[m] += 1  # Fill accuracy bin
        
#         # If this bin is not empty calculate accuracy and confidence
#         if Bm[m] != 0:  
#             acc[m] = acc[m] / Bm[m]
#             conf[m] = conf[m] / Bm[m]

#     # Calculate ece
#     ece = 0

#     # For each bin
#     for m in range(n_bins):
#         ece += Bm[m] * np.abs((acc[m] - conf[m]))
#     return ece / sum(Bm)


def balanced_acc_score(model_output, model_labels):
    result = balanced_accuracy_score(model_labels, model_output)
    result = float(result)
    return result


def auroc_score(model_output, model_labels, n_classes):
    auroc = AUROC(num_classes=n_classes)
    result = auroc(model_output, model_labels)
    result = float(result)
    return result


# def sensitivity_score(model_output, model_labels, n_classes):
#     #https://yeseullee0311.medium.com/pytorch-performance-evaluation-of-a-classification-model-confusion-matrix-fbec6f4e8d0
#     CM = 0
#     preds = torch.argmax(model_output.data, 1)
#     CM += confusion_matrix(model_labels.cpu(), preds.cpu(),labels=np.arange(n_classes))

#     print(CM)

#     tn=CM[0][0]
#     tp=CM[1][1]
#     fp=CM[0][1]
#     fn=CM[1][0]

#     print(tn)
#     print(tp)
#     print(fp)
#     print(fn)
#     acc=np.sum(np.diag(CM)/np.sum(CM))
#     sensitivity=tp/(tp+fn)
#     precision=tp/(tp+fp)
#     specificity = (tp/(tp+fn))

#     # print('\nTestset Accuracy(mean): %f %%' % (100 * acc))
#     # print()
#     # print('Confusion Matirx : ')
#     # print('- Sensitivity : ',(tp/(tp+fn))*100)
#     # print('- Specificity : ',(tn/(tn+fp))*100)
#     # print('- Precision: ',(tp/(tp+fp))*100)
#     # print('- NPV: ',(tn/(tn+fn))*100)
#     # print('- F1 : ',((2*sensitivity*precision)/(sensitivity+precision))*100)
#     # print()
# )
#     return sensitivity, specificity

def specificity_score(model_output, model_labels, n_classes):
    metric =  MulticlassSpecificity(num_classes=n_classes)
    result = metric(model_output, model_labels)
    result = np.array(result)
    result = float(result)
    return result