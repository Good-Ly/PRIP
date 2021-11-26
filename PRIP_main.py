import os
import sklearn
import joblib
import xgboost
import warnings
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from prettytable import PrettyTable
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc,precision_recall_curve,average_precision_score
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
for module in xgboost,np,sklearn:
    print(module.__name__,module.__version__)


def calculateEvaluationMetrics(x_train, y_train,x_test,y_test,pos_label=1):
    predict_pos_proba, predict_label = constructXGBoost(x_train, y_train, x_test,y_test,seed = 75)
    pos_num = np.sum(y_test == pos_label)
    neg_num = y_test.shape[0] - pos_num
    tp = np.sum((y_test == pos_label) & (predict_label == pos_label))
    tn = np.sum(y_test == predict_label) - tp
    sn = (format(tp / pos_num, '.4f'))
    sp = (format(tn / neg_num, '.4f'))
    acc = (format((tp + tn) / (pos_num + neg_num), '.4f'))
    fn = pos_num - tp
    fp = neg_num - tn
    prec = (format(tp / (tp + fp), '.4f'))
    prec_nofloat = tp / (tp + fp)
    rec_nofloat = tp / (tp + fn)
    F_1 = 2 * prec_nofloat * rec_nofloat / (prec_nofloat + rec_nofloat)
    F1 = (round(F_1, 4))
    PRC = average_precision_score(y_test, predict_pos_proba)
    fpr, tpr, _ = roc_curve(y_test, predict_pos_proba)
    roc = auc(fpr, tpr)
    AUPRC = (round(PRC, 4))
    AUROC = (round(roc, 4))
    tp = np.array(tp,dtype=np.float64)
    tn = np.array(tn,dtype=np.float64)
    fp = np.array(fp,dtype=np.float64)
    fn = np.array(fn,dtype=np.float64)
    mcc = (format((tp * tn - fp * fn) / (np.sqrt((tp + fn) * (tp + fp) * (tn + fp) * (tn + fn))), '.4f'))
    precision, recall, thresholds = precision_recall_curve(y_test, predict_pos_proba)
    return sn, sp, prec, acc, F1, mcc, AUROC, AUPRC, fpr, tpr, precision, recall

print("Data loading......")
len_seq = 39
file_name = str(len_seq)
data_word2vec_2 = r"./Semantic features/Train_Test_numpy_data.npz"

train_test_data_word2vec_2 = np.load(data_word2vec_2)

# *************  feature1: word2vec--> 1 word win_** --> __*25  ***************#
train_data_word = train_test_data_word2vec_2['train_data'].reshape(-1, len_seq * 25)
train_label = train_test_data_word2vec_2['train_label']
test_data_word = train_test_data_word2vec_2['test_data'].reshape(-1, len_seq * 25)
test_label = train_test_data_word2vec_2['test_label']

# ***************************************************#
print("train_data_word_2.shape:", train_data_word.shape)
print("train_data_word_2.shape:", test_data_word.shape)
print("train_label.shape:", train_label.shape)
print("test_label.shape:", test_label.shape)

# # *******************#
# standar_scal = False
# min_max_scal = True
#
# if min_max_scal == True:
#     min_max_scaler = preprocessing.MinMaxScaler()
#     train_data = min_max_scaler.fit_transform(train_data_word)
#     test_data = min_max_scaler.fit_transform(test_data_word)
#     print("min_max_scalar")
# elif standar_scal == False:
#     standar_scaler = preprocessing.StandardScaler()
#     train_data = standar_scaler.fit_transform(train_data_word)
#     test_data = standar_scaler.fit_transform(test_data_word)
#     print("standar_scaler")
# else:
train_data = train_data_word
test_data = test_data_word

# ***** shuffle data********#
np.random.seed(200)
np.random.shuffle(train_data)
np.random.seed(200)
np.random.shuffle(train_label)

np.random.seed(200)
np.random.shuffle(test_data)
np.random.seed(200)
np.random.shuffle(test_label)

print("train_data.shape:", train_data.shape)
print("test_data.shape:", test_data.shape)

def constructXGBoost(x_train, y_train, x_test, y_test,seed):
    XGB = XGBClassifier(learning_rate = 0.0311,
                         n_estimators = 331,
                         max_depth = 3,
                         min_child_weight = 1,
                         gamma =0.6,
                         subsample= 1,
                         colsample_bytree= 0.07,
                         objective='binary:logistic',
                         eval_metric=['logloss','auc','error'],
                         #nthread=12,
                         tree_method='gpu_hist',
                         scale_pos_weight = 7,
                         reg_alpha= 2,
                         reg_lambda = 1,
                         seed = 39)
    XGB = XGB.fit( x_train, y_train)
    predict_pos_proba = XGB.predict_proba(x_test)[:, 1]
    predict_label = XGB.predict(x_test)
    # save model
    joblib.dump(XGB, r'./models/5-fold cross-validation/PRIP '+" fold %d.model"%(k_fold_iter_i+1))
    return predict_pos_proba, predict_label
print('****************************')
print("5-fold cross-validation......")
mean_fpr = np.linspace(0, 1, 100)
fpr_sum = []
tpr_sum = []
AUROC_sum = []
recall_sum = []
kf = KFold(n_splits=5)

k_fold_iter_i = 0
sn_list = np.zeros(5)
sp_list = np.zeros(5)
acc_list = np.zeros(5)
mcc_list = np.zeros(5)
pre_list = np.zeros(5)
f1 = np.zeros(5)
auroc = np.zeros(5)
auprc = np.zeros(5)
k_fold_iter_i = 0
for train, test in kf.split(train_data):
    sn, sp, prec, acc, F1, mcc, AUROC, AUPRC, fpr, tpr, precision, recall = calculateEvaluationMetrics(
        train_data[train],
        train_label[train],
        train_data[test],
        train_label[test], pos_label=1)
    sn_list[k_fold_iter_i] = sn
    sp_list[k_fold_iter_i] = sp
    acc_list[k_fold_iter_i] = acc
    mcc_list[k_fold_iter_i] = mcc
    pre_list[k_fold_iter_i] = prec
    f1[k_fold_iter_i] = F1
    auroc[k_fold_iter_i] = AUROC
    auprc[k_fold_iter_i] = AUPRC
    x = PrettyTable(["cross validation", "SN","SP","ACC","MCC","AUROC","Precision","F1","AUPRC"])
    x.add_row(["{}fold".format(k_fold_iter_i+1), sn, sp, acc, mcc, AUROC, prec, F1, AUPRC])
    print(x)
    tpr_sum.append(np.interp(mean_fpr, fpr, tpr))
    tpr_sum[-1][0] = 0.0
    AUROC_sum.append(AUROC)
    mean_tpr = np.mean(tpr_sum, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(AUROC_sum)
    plt.plot(fpr, tpr, lw=1, alpha=0.7, label='ROC fold %d (AUROC = %0.4f)' % (k_fold_iter_i + 1, AUROC))
    k_fold_iter_i = k_fold_iter_i + 1
print('**** mean ****')
print('sn_mean:', np.round(sn_list.mean(), 4))
print("sp_mean:", np.round(sp_list.mean(), 4))
print("acc_mean:", np.round(acc_list.mean(), 4))
print("mcc_mean:", np.round(mcc_list.mean(), 4))
print('prec_mean:', np.round(pre_list.mean(), 4))
print("F1_mean:", np.round(f1.mean(), 4))
print("AUROC_mean:", np.round(auroc.mean(), 4))
print("AUPRC_mean:", np.round(auprc.mean(), 4))

# Drawing the ROC curve of the 5-fold cross validation
print("*********************************************************")
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# plt.title('ROC Curves')
plt.legend(loc="lower right", prop={"size": 6})
plt.savefig(r"./Figure/5-fold cross-validation/PRIP 5-fold cross-validation ROC Curves.png",dpi=300)
plt.savefig(r"./Figure/5-fold cross-validation/PRIP 5-fold cross-validation ROC Curves.pdf")
plt.show()  #

print("Independent testing")
def constructXGBoost(x_train, y_train, x_test, y_test,seed):
    XGB = XGBClassifier(learning_rate = 0.0311,
                         n_estimators = 331,
                         max_depth = 3,
                         min_child_weight = 1,
                         gamma =0.6,
                         subsample= 1,
                         colsample_bytree= 0.07,
                         objective='binary:logistic',
                         eval_metric=['logloss','auc','error'],
                         #nthread=12,
                         tree_method='gpu_hist',
                         scale_pos_weight = 7,
                         reg_alpha= 2,
                         reg_lambda = 1,
                         seed = 39)
    XGB = XGB.fit( x_train, y_train)
    predict_pos_proba = XGB.predict_proba(x_test)[:, 1]
    predict_label = XGB.predict(x_test)
    # save model
    joblib.dump(XGB, r'./models/Independent testing/Independent testing .model')
    return predict_pos_proba, predict_label


mean_fpr=np.linspace(0,1,100)
fpr_sum = []
tpr_sum = []
AUROC_sum = []
recall_sum = []
print("******************  Independence test  ******************")
sn, sp, prec, acc, F1, mcc, AUROC, AUPRC, fpr, tpr, precision, recall = calculateEvaluationMetrics(train_data,
                                                                                                   train_label,
                                                                                                   test_data,
                                                                                                   test_label,pos_label=1)
x = PrettyTable(["SN", "SP","ACC","MCC", "AUROC","Precision","F1","AUPRC"])
x.add_row([sn, sp,acc, mcc,AUROC, prec, F1,  AUPRC])
print(x)
tpr_sum.append(np.interp(mean_fpr, fpr, tpr))
tpr_sum[-1][0] = 0.0
AUROC_sum.append(AUROC)
mean_tpr = np.mean(tpr_sum, axis=0)
mean_tpr[-1] = 1.0
plt.plot(mean_fpr, mean_tpr,label=r'PRIP ROC(AUROC = %0.4f )' %(AUROC),lw=2, alpha=.8)
# Drawing the ROC curve of the 5-fold cross validation
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Chance', alpha=.8)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title('ROC Curve')
plt.legend(loc="lower right", prop={"size": 6})
plt.savefig(r"./Figure/independence test/PRIP ROC Curve：independence test.png",dpi = 300)
plt.savefig(r"./Figure/independence test/PRIP ROC Curve：independence test.png.pdf")
plt.show()
