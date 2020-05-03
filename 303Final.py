import numpy as np
import pandas as pd
import csv
import json
import xml.etree.ElementTree as et



#reading in text file
handle = open('Loans.txt', 'r')
reader = csv.reader(handle, delimiter='|')
rows = list(reader)
#it kept reading in the data as objects creating the dataframe from rows, which is a problem when I need to merge. This is why I used read_csv
my_df = pd.read_csv('Loans.txt', sep='|')
my_df.columns = rows[0]

#reading in json
handle = open('Customers.json', 'r')
json_loaded = json.load(handle)
pdjson = pd.DataFrame.from_dict(json_loaded['customers'], dtype='int64')

#merging text and json into one dataframe
merged = my_df.merge(pdjson, on='CustomerID', how='inner')

#reading in xml file
xtree = et.parse('Payments.xml')
xroot = xtree.getroot()
my_list=[]
for node in xroot.findall('./payment'):
    loan_num = None
    pay_amount = None
    for child in node:
        my_dict = {}
        if child.tag == 'loannumber':
            loan_num = child.text if child is not None else None
        if child.tag == 'paymentamount':
            pay_amount = child.text if child is not None else None
        if loan_num != None and pay_amount != None:
            my_dict.update({'loannumber': loan_num, 'paymentamount': float(pay_amount)})
            my_list.append(my_dict)


xml_df = pd.DataFrame(my_list)

#merge xml with other data
merged = pd.merge(merged, xml_df, left_on='loannumber', right_on='loannumber', how='left')


#convert unpaid invoices to 0's, paid invoices to 1's
merged['paymentamount'].fillna(0, inplace=True)
merged['paymentamount'] = merged['paymentamount'].apply(lambda x: 1 if x > 0 else 0)

#encode income verification status to 1 being verified, 0 being not verified
merged['inc_verification_status'] = merged['inc_verification_status'].apply(lambda x: ['Not Verified', 'Verified'].index(x))


#binarize emp_category: 1 = white collar, 0 = blue collar
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
lb.fit(merged['emp_category'].to_numpy().reshape(merged.shape[0], 1))
merged['emp_category']= lb.transform(merged['emp_category'].to_numpy().reshape(merged.shape[0], 1))

#use onehot encoder to encode home ownership variable
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
le = LabelEncoder()
ownership_names = ['OWN', 'MORTGAGE', 'RENT']
le.fit(ownership_names)
trans = merged['home_ownership'].to_numpy().reshape(merged.shape[0],1)
le_data = le.transform(trans).reshape(merged.shape[0],1)
ohe = OneHotEncoder(sparse=False)
fr_reshape = merged['home_ownership'].to_numpy().reshape(merged.shape[0],1)
ohefrg = ohe.fit_transform(le_data)
merged['home_ownership'] = ohefrg

#scale continuous variables
from sklearn.preprocessing import StandardScaler
cont_features = merged[['OriginalLoanAmount', 'DelinquentAmount', 'LoanTerm','LoanInterestRate','MonthlyPayment','NumberOnTimePayments',
                       'NumberPreviousLatePayments', 'CustomerCreditScore', 'emp_length', 'annual_inc' ,'NumberCreditInquriesLast12Months',
                       'OpenLinesOfCredit', 'CreditUtilization','pub_rec','Debt_to_Income_Ratio']].values
sc = StandardScaler().fit(cont_features)
cont_sc = sc.transform(cont_features)

#put continuous variables back into a dataframe with the non scaled categorical variables, removing loan number and customer ID
merged2 = pd.DataFrame(cont_sc, columns=['OriginalLoanAmount', 'DelinquentAmount', 'LoanTerm','LoanInterestRate','MonthlyPayment','NumberOnTimePayments',
                       'NumberPreviousLatePayments', 'CustomerCreditScore', 'emp_length', 'annual_inc' ,'NumberCreditInquriesLast12Months',
                       'OpenLinesOfCredit', 'CreditUtilization','pub_rec','Debt_to_Income_Ratio'])
merged2['LoanStatus'] = merged['LoanStatus']
merged2['home_ownership'] = merged['home_ownership']
merged2['emp_category'] = merged['emp_category']
merged2['inc_verification_status'] = merged['inc_verification_status']
merged2['paymentamount'] = merged['paymentamount']

#fit the pca, split the data and fit the data to a logistic regression model
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
pca = PCA(n_components=2, random_state=23)
features = merged2.values
features_reduced = pca.fit_transform(features)

f_train, f_test, l_train, l_test = train_test_split(features_reduced, merged2['paymentamount'], test_size=0.5, random_state=23)
projects_model = LogisticRegression(C=1, random_state=23)


#display model equation (number 5) and variance that uncorrelated dimensions have on the total variance (number 3)
vars = pca.explained_variance_ratio_
c_names = ['OriginalLoanAmount', 'DelinquentAmount', 'LoanTerm','LoanInterestRate','MonthlyPayment','NumberOnTimePayments',
                       'NumberPreviousLatePayments', 'CustomerCreditScore', 'emp_length', 'annual_inc' ,'NumberCreditInquriesLast12Months',
                       'OpenLinesOfCredit', 'CreditUtilization','pub_rec','Debt_to_Income_Ratio','paymentamount', 'LoanStatus', 'home_ownership',
          'emp_category', 'inc_verification_status']
print('Variance:  Projected dimension')
print('------------------------------')
for idx, row in enumerate(pca.components_):
    output = '{0:4.1f}%:    '.format(100.0 * vars[idx])
    output += " + ".join("{0:5.2f} * {1:s}".format(val, name) \
                      for val, name in zip(row, c_names))
    print(output)


# Fit estimator and display score (number 6)
trained_model = projects_model.fit(f_train, l_train)
predicted = trained_model.predict(f_test)
print('Score = {:.1%}'.format(projects_model.score(f_test, l_test)))

#Number 7, display confusion matrix (number 7)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import seaborn as sns

print('Computed metrics from the scikit-learn module')
print(45*'-')
print(f'Precision   = {100.0 * precision_score(l_test, predicted):5.2f}%')
print(f'Accuracy    = {100.0 * accuracy_score(l_test, predicted):5.2f}%')
print(f'Recall      = {100.0 * recall_score(l_test, predicted):5.2f}%')
print(f'F1-score    = {100.0 * f1_score(l_test, predicted):5.2f}%')

def confusion(test, predict, title, labels):
    pts, xe, ye = np.histogram2d(test, predict, bins=2)

    pd_pts = pd.DataFrame(pts.astype(int), index=labels, columns=labels)

    hm = sns.heatmap(pd_pts, annot=True, fmt="d")
    hm.axes.set_title(title, fontsize=20)
    hm.axes.set_xlabel('True Label', fontsize=18)
    hm.axes.set_ylabel('Predicted Label', fontsize=18)

    return None

confusion(l_test, predicted, f'Predicting Payment on Loan', ['Pay', 'Unpaid'])

#create an roc curve (number 7)
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
fpr = dict()
tpr = dict()
roc_auc = dict()
y_score = projects_model.fit(f_train, l_train).decision_function(f_test)
fpr[0], tpr[0], _ = roc_curve(l_test, y_score)
roc_auc[0] = auc(fpr[0], tpr[0])
print(roc_auc[0])

fpr["micro"], tpr["micro"], _ = roc_curve(l_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
lw = 2
plt.plot(fpr[0], tpr[0], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
