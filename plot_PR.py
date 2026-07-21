from sklearn import svm, datasets
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score, roc_curve, auc, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
#from matplotlib.pyplot import MultipleLocater
#from sklearn.utils.fixes import signature

import csv
import numpy as np

#the csv file of semi-supervised BC dataset

bs2058_csv_name1 = 'flexmatch_bus_878_cv1.csv'
bs2058_csv_name2 = 'flexmatch_bus_878_cv2.csv'
bs2058_csv_name3 = 'flexmatch_bus_878_cv3.csv'
bs2058_csv_name4 = 'flexmatch_bus_878_cv4.csv'
bs2058_csv_name5 = 'flexmatch_bus_878_cv5.csv'

# calculate F1 score for each csv

# print("###################")

csvfilelist = [bs2058_csv_name1, bs2058_csv_name2, bs2058_csv_name3, bs2058_csv_name4, bs2058_csv_name5]#,
# bs2058_csv_name6, bs2058_csv_name7, bs2058_csv_name8, bs2058_csv_name9, bs2058_csv_name10]
# csvfilelist = [bs2058_csv_name6, bs2058_csv_name7, bs2058_csv_name8, bs2058_csv_name9, bs2058_csv_name10]

plt.figure(figsize=(7, 7))
AUC_list = []
ACC_list = []
Sen_ist = []
Spe_list = []
Fscore_list = []

for name in csvfilelist:
    csvreader = csv.reader(open(name, 'r'))
    csvlist = list(csvreader)
    label = []
    prob0 = []
    prob1 = []
    pre_label = []
    mcount = 0
    bcount = 0
    mcorrect = 0
    bcorrect = 0
    precision = dict()
    recall = dict()
    average_precision = ()
    # n_classes = []
    for i in range(1, len(csvlist)):
        label.append(int(csvlist[i][1]))
        prob0.append(float(csvlist[i][3]))
        prob1.append(float(csvlist[i][4]))
        if int(csvlist[i][1]) == 0:
            bcount += 1
        else:
            mcount += 1

        if float(csvlist[i][3]) > float(csvlist[i][4]):
            pre_label.append(0)
            if int(csvlist[i][1]) == 0:
                bcorrect += 1
        else:
            pre_label.append(1)
            if int(csvlist[i][1]) == 1:
                mcorrect += 1


    fscore = f1_score(label,pre_label)

    Fscore_list.append(fscore)
    ACC_list.append(float((bcorrect + mcorrect)/ (bcount+mcount)))
    Sen_ist.append(float(mcorrect / mcount))
    Spe_list.append(float(bcorrect / bcount))
    print("$$$$$$$$$$$$$$$$$$$$$")
    print("Fscore***********")
    print(Fscore_list)
    print("Fscore mean and std is %f+-%f"%(np.mean(Fscore_list),np.std(Fscore_list)))
    print("ACC***********")
    print(ACC_list)
    print("ACC mean and std is %f+-%f"%(np.mean(ACC_list),np.std(ACC_list)))
    print("Sen***********")
    print(Sen_ist)
    print("Sen mean and std is %f+-%f"%(np.mean(Sen_ist),np.std(Sen_ist)))
    print("Spe***********")
    print(Spe_list)
    print("Spe mean and std is %f+-%f"%(np.mean(Spe_list),np.std(Spe_list)))

    average_precision = average_precision_score(label, prob1)
    fpr, tpr, thresholds = roc_curve(label, prob1)
    roc_auc = auc(fpr, tpr)
    AUC_list.append(roc_auc)
    print("AUC***********")
    print(AUC_list)
    print("The AUC is %f"%(roc_auc))
    print("AUC mean and std is %f+-%f"%(np.mean(AUC_list),np.std(AUC_list)))
    
    print('Average precision-recall score: {0:0.4f}'.format(
        average_precision))
    precision, recall, _ = precision_recall_curve(label, prob1)

