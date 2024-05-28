import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, accuracy_score, precision_score,average_precision_score
import os
import pandas as pd
import torch
import itertools
import numpy as np



def classification_metrics_softmax(label, prediction):
    prediction_prob = prediction[:, 1]
    fpr, tpr, threshold = metrics.roc_curve(label, prediction_prob, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.abs(fnr - fpr))]
    eer = fpr[np.nanargmin(np.abs(fnr - fpr))]

    auc = roc_auc_score(label, prediction_prob)
    prediction_labels = np.argmax(prediction, axis=1)
    acc = accuracy_score(label, prediction_labels)
    precision = average_precision_score(label, prediction_prob)

    CM = confusion_matrix(label, prediction_labels)
    TN, FP, FN, TP = CM.ravel()

    FPR = FP / (FP + TN)
    TPR = TP / (TP + FN)
    FNR = FN / (FN + TP)
    TNR = TN / (TN + FP)
    f_g = (TP + FP) / len(label)
    positive_rate = f_g
    negative_rate = (TN + FN) / len(label)
    er = 1 - ((TPR + (1 - FPR)) / 2)

    return auc, er, FPR, TPR, acc, precision, f_g, eer, positive_rate, negative_rate, FNR, TNR


def softmax(input_array, dim=1):
    input_array = input_array - np.max(input_array, axis=dim, keepdims=True)

    exp_values = np.exp(input_array)

    sum_exp_values = np.sum(exp_values, axis=dim, keepdims=True)

    softmax_output = exp_values / sum_exp_values
   
    return softmax_output

def acc_fairness_softmax(path, attribute_list):
    dict_data = {}

    dict_data['group_1'] = {}
    dict_data['group_1']['label'] = []
    dict_data['group_1']['logits'] = []
    dict_data['group_1']['prediction'] = []

    dict_data['group_2'] = {}
    dict_data['group_2']['label'] = []
    dict_data['group_2']['logits'] = []
    dict_data['group_2']['prediction'] = []

    dict_data['group_3'] = {}
    dict_data['group_3']['label'] = []
    dict_data['group_3']['logits'] = []
    dict_data['group_3']['prediction'] = []

    length = len(attribute_list)
    if length == 3:
        for g1att in attribute_list[0]:
            if g1att not in dict_data.keys():
                dict_data[g1att] = {}
                dict_data[g1att]['label'] = []
                dict_data[g1att]['logits'] = []
                dict_data[g1att]['prediction'] = []
            for g2att in attribute_list[1]:
                if g2att not in dict_data.keys():
                    dict_data[g2att] = {}
                    dict_data[g2att]['label'] = []
                    dict_data[g2att]['logits'] = []
                    dict_data[g2att]['prediction'] = []
                label_path = path+g1att+','+g2att+"labels.npy"
                print(label_path, '')
                prediction_path = path+g1att+','+g2att+"predictions.npy"
                label = np.load(label_path)

                prediction = np.load(prediction_path)


                if len(label) != 0:

                    dict_data[g1att+','+g2att] = {}
                    dict_data[g1att+','+g2att]['label'] = label
                    dict_data[g1att+','+g2att]['logits'] = prediction
                    dict_data[g1att+',' +
                              g2att]['prediction'] = softmax(prediction)
                    dict_data[g1att]['label'] += (list(label))
                    dict_data[g1att]['logits'] += (list(prediction))
                    dict_data[g1att]['prediction'] += list(softmax(prediction))
                    dict_data[g2att]['label'] += (list(label))
                    dict_data[g2att]['logits'] += (list(prediction))
                    dict_data[g2att]['prediction'] += list(softmax(prediction))
                    dict_data['group_1']['label'] += (list(label))
                    dict_data['group_1']['logits'] += (list(prediction))
                    dict_data['group_1']['prediction'] += (
                        list(softmax(prediction)))
                    dict_data['group_2']['label'] += (list(label))
                    dict_data['group_2']['logits'] += (list(prediction))
                    dict_data['group_2']['prediction'] += (
                        list(softmax(prediction)))
                    
        for g3att in attribute_list[2]:
            if g3att not in dict_data.keys():
                dict_data[g3att] = {}
                dict_data[g3att]['label'] = []
                dict_data[g3att]['logits'] = []
                dict_data[g3att]['prediction'] = []

                label_path = path+g3att+"labels.npy"
                print(label_path, '')
                prediction_path = path+g3att+"predictions.npy"
                label = np.load(label_path)

                prediction = np.load(prediction_path)
                if len(label) != 0:
                    dict_data[g3att] = {}
                    dict_data[g3att]['label'] = label
                    dict_data[g3att]['logits'] = prediction
                    dict_data[g3att]['prediction'] = softmax(prediction)
                    dict_data['group_3']['label'] += (list(label))
                    dict_data['group_3']['logits'] += (list(prediction))
                    dict_data['group_3']['prediction'] += (
                        list(softmax(prediction)))
                    
    # print(dict_data.keys())


    metrics = {}
    for item in dict_data.keys():
        # print(item, '1111111')
        metrics[item] = {}
        totalnumber = len(dict_data[item]['label'])
        if totalnumber != 0:
            try:
                # print('!!!!!!!!!!!!!!!!!!!!!!!')
                auc, ErrorRate, FPR, TPR, acc, precision, F_G, eer, positive_rate, negative_rate, FNR, TNR = classification_metrics_softmax(
                    np.array(dict_data[item]['label']), np.array(dict_data[item]['prediction']))
                metrics[item]['auc'] = auc
                metrics[item]['ErrorRate'] = ErrorRate
                metrics[item]['FPR'] = FPR
                metrics[item]['TPR'] = TPR
                metrics[item]['acc'] = acc
                metrics[item]['precision'] = precision
                metrics[item]['F_G'] = F_G
                metrics[item]['eer'] = eer
                metrics[item]['positive_rate'] = positive_rate
                metrics[item]['negative_rate'] = negative_rate
                metrics[item]['FNR'] = FNR
                metrics[item]['TNR'] = TNR

                print(f' number of {totalnumber} with {item} : AUC: {auc:.5f} | ErrorRate: {ErrorRate: .5f} | fpr: {FPR:.5f} | tpr:{TPR:.5f} | acc: {acc:.5f} | precision: {precision:.5f} | EER: {eer:.5f} | F_G: {F_G:.5f}')
            except:
                print(
                    f' no positive or negative samples in the {totalnumber} of {item}, can not output the metrics')
        else:
            print(f' number of {item} is empty')
            continue

    return_results = []
    # print(metrics)

    bbb = 0.0
    aaa = []
    efpr_inter = 0.0
    b_inter = 0.0
    a_inter = 0.0
    for names in metrics.keys():
        # print(metrics.keys(), 'metrics.keys()')

        if len(names.split(',')) >= 2 and metrics[names] != {}:
            if a_inter < np.abs(metrics[names]['F_G']-metrics['group_1']['F_G']):
                a_inter = np.abs(
                    metrics[names]['F_G']-metrics['group_1']['F_G'])

            b_inter += (np.abs(metrics[names]['FPR']-metrics['group_1']['FPR'])+np.abs(
                metrics[names]['TPR']-metrics['group_1']['TPR']))
            efpr_inter += (np.abs(metrics[names]
                           ['FPR']-metrics['group_1']['FPR']))

            aaa.append(metrics[names]['F_G'])

    for i in range(len(aaa)):
        for j in range(i+1, len(aaa)):
            if bbb < (np.abs(aaa[i]-aaa[j])):
                bbb = (np.abs(aaa[i]-aaa[j]))
    print(f' F_S: {bbb:.5f} ')
    F_A_inter = (bbb + a_inter) / 2
    print(
        f' F_G_inter, F_EFPR_inter, F_EO_inter, F_A_inter: {a_inter:.5f} | {efpr_inter:.5f} | {b_inter:.5f} | {F_A_inter:.5f} ')

    return_results += [metrics['group_1']['auc'], metrics['group_1']['FPR'],
                       metrics['group_1']['TPR'], metrics['group_1']['acc'], efpr_inter, b_inter]

    gap_g1, gap_g2, gap_g3,  gap_inter, = {}, {}, {},{}
    gap_g1['auc'], gap_g1['ErrorRate'], gap_g1['FPR'], gap_g1['acc'], gap_g1['foae'], gap_g1['PR'], gap_g1['NR'], gap_g1['Fdp_diff'], gap_g1['F_DP'], gap_g1['FNR'], gap_g1['TNR'], gap_g1['TPR'] = [
    ], [], [], [], [], [], [], [], [], [], [], []
    gap_g2['auc'], gap_g2['ErrorRate'], gap_g2['FPR'], gap_g2['acc'], gap_g2['foae'], gap_g2['PR'], gap_g2['NR'], gap_g2['Fdp_diff'], gap_g2['F_DP'], gap_g2['FNR'], gap_g2['TNR'], gap_g2['TPR'] = [
    ], [], [], [], [], [], [], [], [], [], [], []
    gap_g3['auc'], gap_g3['ErrorRate'], gap_g3['FPR'], gap_g3['acc'], gap_g3['foae'], gap_g3['PR'], gap_g3['NR'], gap_g3['Fdp_diff'], gap_g3['F_DP'], gap_g3['FNR'], gap_g3['TNR'], gap_g3['TPR'] = [
    ], [], [], [], [], [], [], [], [], [], [], []
    gap_inter['auc'], gap_inter['ErrorRate'], gap_inter['FPR'], gap_inter['acc'], gap_inter['foae'], gap_inter['PR'], gap_inter['NR'], gap_inter['Fdp_diff'], gap_inter['F_DP'], gap_inter['FNR'], gap_inter['TNR'], gap_inter['TPR'] = [
    ], [], [], [], [], [], [], [], [], [], [], []
    gap_g1['Fmeo_diff'], gap_g2['Fmeo_diff'], gap_g3['Fmeo_diff'], gap_inter['Fmeo_diff'] = [], [], [], []
    for index, group in enumerate(attribute_list[:3]):
        a = 0.0
        b = 0.0
        efpr = 0.0
        cov = 0.0
        for weight, item in enumerate(group):
            if metrics[item] != {}:
                if a < np.abs(metrics[item]['F_G']-metrics['group_'+str(index+1)]['F_G']):
                    a = np.abs(metrics[item]['F_G'] -
                               metrics['group_'+str(index+1)]['F_G'])

                b += (np.abs(metrics[item]['FPR']-metrics['group_'+str(index+1)]['FPR'])+np.abs(
                    metrics[item]['TPR']-metrics['group_'+str(index+1)]['TPR']))
                efpr += np.abs(metrics[item]['FPR'] -
                               metrics['group_'+str(index+1)]['FPR'])
            if dict_data[item] != {}:
                firstitem = np.sum(
                    weight*dict_data[item]['logits'])/len(dict_data['group_'+str(index+1)]['label'])
                seconditem = np.multiply((weight*len(dict_data[item]['label'])/len(dict_data['group_'+str(
                    index+1)]['label'])), np.mean(dict_data['group_'+str(index+1)]['logits']))
                cov += np.abs(firstitem-seconditem)
                # print(cov)
        ccc = (a + bbb)/2
        print(
            f' F_G, F_EFPR, F_EO, F_A, F_COV of group_{index+1}: {a:.5f} | {efpr:.5f} | {b:.5f} | {ccc:.5f} | {cov:.5f}')
        return_results += [efpr, b]

    for g1 in attribute_list[0]:
        try:
            gap_g1['auc'].append(metrics[g1]['auc'])
            gap_g1['ErrorRate'].append(metrics[g1]['ErrorRate'])
            gap_g1['FPR'].append(metrics[g1]['FPR'])
            gap_g1['acc'].append(metrics[g1]['acc'])
            gap_g1['foae'].append(metrics[g1]['acc'])
            gap_g1['PR'].append(metrics[g1]['positive_rate'])
            gap_g1['NR'].append(metrics[g1]['negative_rate'])
            gap_g1['FNR'].append(metrics[g1]['FNR'])
            gap_g1['TNR'].append(metrics[g1]['TNR'])
            gap_g1['TPR'].append(metrics[g1]['TPR'])

        except:
            continue
    gap_g1['Fdp_diff'].append(max(gap_g1['PR'])-min(gap_g1['PR']))
    gap_g1['Fdp_diff'].append(max(gap_g1['NR'])-min(gap_g1['NR']))
    gap_g1['Fmeo_diff'].append(max(gap_g1['FPR'])-min(gap_g1['FPR']))
    gap_g1['Fmeo_diff'].append(max(gap_g1['FNR'])-min(gap_g1['FNR']))
    gap_g1['Fmeo_diff'].append(max(gap_g1['TNR'])-min(gap_g1['TNR']))
    gap_g1['Fmeo_diff'].append(max(gap_g1['TPR'])-min(gap_g1['TPR']))

    for g2 in attribute_list[1]:
        try:
            gap_g2['auc'].append(metrics[g2]['auc'])
            gap_g2['ErrorRate'].append(metrics[g2]['ErrorRate'])
            gap_g2['FPR'].append(metrics[g2]['FPR'])
            gap_g2['acc'].append(metrics[g2]['acc'])
            gap_g2['foae'].append(metrics[g2]['acc'])
            gap_g2['PR'].append(metrics[g2]['positive_rate'])
            gap_g2['NR'].append(metrics[g2]['negative_rate'])
            gap_g2['FNR'].append(metrics[g2]['FNR'])
            gap_g2['TNR'].append(metrics[g2]['TNR'])
            gap_g2['TPR'].append(metrics[g2]['TPR'])
        except:
            continue
    gap_g2['Fdp_diff'].append(max(gap_g2['PR'])-min(gap_g2['PR']))
    gap_g2['Fdp_diff'].append(max(gap_g2['NR'])-min(gap_g2['NR']))
    gap_g2['Fmeo_diff'].append(max(gap_g2['FPR'])-min(gap_g2['FPR']))
    gap_g2['Fmeo_diff'].append(max(gap_g2['FNR'])-min(gap_g2['FNR']))
    gap_g2['Fmeo_diff'].append(max(gap_g2['TNR'])-min(gap_g2['TNR']))
    gap_g2['Fmeo_diff'].append(max(gap_g2['TPR'])-min(gap_g2['TPR']))

    for g3 in attribute_list[2]:
        try:
            gap_g3['auc'].append(metrics[g3]['auc'])
            gap_g3['ErrorRate'].append(metrics[g3]['ErrorRate'])
            gap_g3['FPR'].append(metrics[g3]['FPR'])
            gap_g3['acc'].append(metrics[g3]['acc'])
            gap_g3['foae'].append(metrics[g3]['acc'])
            gap_g3['PR'].append(metrics[g3]['positive_rate'])
            gap_g3['NR'].append(metrics[g3]['negative_rate'])
            gap_g3['FNR'].append(metrics[g3]['FNR'])
            gap_g3['TNR'].append(metrics[g3]['TNR'])
            gap_g3['TPR'].append(metrics[g3]['TPR'])
        except:
            continue

    gap_g3['Fdp_diff'].append(max(gap_g3['PR'])-min(gap_g3['PR']))
    gap_g3['Fdp_diff'].append(max(gap_g3['NR'])-min(gap_g3['NR']))
    gap_g3['Fmeo_diff'].append(max(gap_g3['FPR'])-min(gap_g3['FPR']))
    gap_g3['Fmeo_diff'].append(max(gap_g3['FNR'])-min(gap_g3['FNR']))
    gap_g3['Fmeo_diff'].append(max(gap_g3['TNR'])-min(gap_g3['TNR']))
    gap_g3['Fmeo_diff'].append(max(gap_g3['TPR'])-min(gap_g3['TPR']))

    for inter1 in attribute_list[0]:
        for inter2 in attribute_list[1]:
            try:
                gap_inter['auc'].append(metrics[inter1+','+inter2]['auc'])
                gap_inter['ErrorRate'].append(
                    metrics[inter1+','+inter2]['ErrorRate'])
                gap_inter['FPR'].append(metrics[inter1+','+inter2]['FPR'])
                gap_inter['acc'].append(metrics[inter1+','+inter2]['acc'])
                gap_inter['foae'].append(metrics[inter1+','+inter2]['acc'])
                gap_inter['PR'].append(
                    metrics[inter1+','+inter2]['positive_rate'])
                gap_inter['NR'].append(
                    metrics[inter1+','+inter2]['negative_rate'])
                gap_inter['FNR'].append(metrics[inter1+','+inter2]['FNR'])
                gap_inter['TNR'].append(metrics[inter1+','+inter2]['TNR'])
                gap_inter['TPR'].append(metrics[inter1+','+inter2]['TPR'])
            except:
                continue
    gap_inter['Fdp_diff'].append(max(gap_inter['PR'])-min(gap_inter['PR']))
    gap_inter['Fdp_diff'].append(max(gap_inter['NR'])-min(gap_inter['NR']))
    gap_inter['Fmeo_diff'].append(max(gap_inter['FPR'])-min(gap_inter['FPR']))
    gap_inter['Fmeo_diff'].append(max(gap_inter['FNR'])-min(gap_inter['FNR']))
    gap_inter['Fmeo_diff'].append(max(gap_inter['TNR'])-min(gap_inter['TNR']))
    gap_inter['Fmeo_diff'].append(max(gap_inter['TPR'])-min(gap_inter['TPR']))
    for i in ['FPR', 'auc', 'ErrorRate', 'foae']:
        try:
            print(
                f' inter_{i}, g1_{i}, g2_{i}, g3_{i}: {(100*(max(gap_inter[i])-min(gap_inter[i]))):.3f} | {(100*(max(gap_g1[i])-min(gap_g1[i]))):.3f} | {(100*(max(gap_g2[i])-min(gap_g2[i]))):.3f} | {(100*(max(gap_g3[i])-min(gap_g3[i]))):.3f}')
            return_results += [(100*(max(gap_inter[i])-min(gap_inter[i]))), (100*(
                max(gap_g1[i])-min(gap_g1[i]))), (100*(max(gap_g2[i])-min(gap_g2[i]))), (100*(max(gap_g3[i])-min(gap_g3[i])))]
        except:
            continue
    print(
        f' inter_F_DP,g1_F_DP,g2_F_DP,g3_F_DP:{100*max(gap_inter["Fdp_diff"]):.3f} | {100*max(gap_g1["Fdp_diff"]):.3f} | {100*max(gap_g2["Fdp_diff"]):.3f} | {100*max(gap_g3["Fdp_diff"]):.3f}')
    print(
        f' inter_F_meo,g1_F_meo,g2_F_meo, g3_F_meo:{100*max(gap_inter["Fmeo_diff"]):.3f} | {100*max(gap_g1["Fmeo_diff"]):.3f} | {100*max(gap_g2["Fmeo_diff"]):.3f} | {100*max(gap_g3["Fmeo_diff"]):.3f}')
    return return_results

