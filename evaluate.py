import pandas as pd
from sklearn.metrics import average_precision_score

def evaluate(submission_file_name='submission.csv',
             label_file_name='label_test.csv'):
    
    #Submission file from participants
    submission_file_name='submission.csv'
    submission=pd.read_csv(submission_file_name)
    #Sanity check on header
    assert list(submission.columns)==['USUBJID','rapid_progressor']
    
    
    #Label file from Yejin
    label_file_name='label_test.csv'
    label_test=pd.read_csv(label_file_name)
    #Sanity check on header
    assert list(label_test.columns)==['USUBJID','rapid_progressor']
    
    #Assert the number of patients are the same in label_test, submission,
    assert len(label_test)==len(submission)

    #Merge the label and estimated score based on index
    label_test.set_index('USUBJID',inplace=True)
    submission.set_index('USUBJID',inplace=True)
    tmp=pd.merge(label_test, submission,left_index=True, right_index=True, suffixes=('_true','_est'))#Assert label_test and submission had the same patients
    assert len(tmp)==len(label_test)
    
    #Evaluate AUPRC
    auprc=average_precision_score(tmp['rapid_progressor_true'],tmp['rapid_progressor_est'])
    return auprc
    
print(evaluate('submission.csv', 'label_test.csv'))