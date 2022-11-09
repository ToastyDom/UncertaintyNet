#https://github.com/sirius8050/Expected-Calibration-Error/blob/master/ECE.model_output


import numpy as np
import torch
from torchmetrics import AUROC #https://torchmetrics.readthedocs.io/en/stable/classification/auroc.html
from torchmetrics.classification import MulticlassCalibrationError
from torchmetrics.classification import MulticlassSpecificity
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.preprocessing import label_binarize
import calibration


# def ece_score(model_output, model_labels, num_classes, n_bins=10):
#     metric = MulticlassCalibrationError(num_classes=num_classes, n_bins=n_bins, norm='l1')
#     result = metric(model_output, model_labels)
#     result = np.array(result)
#     result = float(result)
#     return result


def ece_score(y_prob, model_labels):

    ece = calibration.get_ece(y_prob, model_labels)
    return ece


def calibration_error(y_prob, model_labels):
    calib_error = calibration.get_calibration_error(y_prob, model_labels)
    return calib_error

def top_calibration_error(y_prob, model_labels):
    top_calib_error = calibration.get_top_calibration_error(y_prob, model_labels, p=1)
    return top_calib_error


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