import numpy as np
import pandas as pd

def engineer_features(d):
    d = d.copy()

    d['Pass_Rate_Sem1'] = (d['Curricular units 1st sem (approved)'] / d['Curricular units 1st sem (enrolled)'].replace(0, np.nan)).fillna(0)
    d['Pass_Rate_Sem2'] = (d['Curricular units 2nd sem (approved)'] / d['Curricular units 2nd sem (enrolled)'].replace(0, np.nan)).fillna(0)
    d['Academic_Momentum']  = d['Pass_Rate_Sem1'] * d['Pass_Rate_Sem2']
    d['Academic_Fatigue']   = d['Pass_Rate_Sem2'] - d['Pass_Rate_Sem1']
    d['Grade_Trend']        = (d['Curricular units 2nd sem (grade)'] - d['Curricular units 1st sem (grade)'])
    d['Avg_Grade']          = (d['Curricular units 1st sem (grade)'] + d['Curricular units 2nd sem (grade)']) / 2
    
    total_enr  = (d['Curricular units 1st sem (enrolled)'] + d['Curricular units 2nd sem (enrolled)'])
    total_appr = (d['Curricular units 1st sem (approved)'] + d['Curricular units 2nd sem (approved)'])
    d['Total_Approved']    = total_appr
    d['Overall_Pass_Rate'] = (total_appr / total_enr.replace(0, np.nan)).fillna(0)
    d['Load_Change']       = (d['Curricular units 2nd sem (enrolled)'] - d['Curricular units 1st sem (enrolled)'])


    d['Zero_Activity_Sem2'] = (d['Curricular units 2nd sem (enrolled)'] == 0).astype(int)
    d['Financial_Stress']   = (d['Debtor'] + (1 - d['Tuition fees up to date']) + (1 - d['Scholarship holder']))
    d['Dropout_Risk']       = (d['Zero_Activity_Sem2'] + d['Financial_Stress'] + (1 - d['Pass_Rate_Sem2']))

    d['Credit_Completion']  = (total_appr / total_enr.replace(0, np.nan)).fillna(0)
    d['Grade_x_PassRate']   = d['Avg_Grade'] * d['Overall_Pass_Rate']
    d['Sem2_Intensity']     = (d['Curricular units 2nd sem (enrolled)'] * d['Pass_Rate_Sem2'])

    d['Enrollment_Stability'] = (1 - abs(d['Academic_Fatigue']))
    d['Active_No_Graduate']   = ((d['Curricular units 2nd sem (enrolled)'] > 0).astype(int) * (1 - d['Zero_Activity_Sem2']))
    return d