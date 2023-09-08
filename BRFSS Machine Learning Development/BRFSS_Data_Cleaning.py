import pandas as pd
import numpy as np

import random

# Import the data from Local Storage
# document explaining the meaning of each column header:
# - https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf

BRFSS_raw_df = pd.read_csv('Data/BRFSS/2015.csv')

# Inspect the raw dataset

print(BRFSS_raw_df.head())
print(BRFSS_raw_df.shape)

# Reduce the dataframe to only contain the desired columns

diabetes_df = BRFSS_raw_df[['DIABETE3',
                            '_RFHYPE5',
                            'TOLDHI2', '_CHOLCHK',
                            '_BMI5',
                            'SMOKE100',
                            'CVDSTRK3', '_MICHD',
                            '_TOTINDA',
                            '_FRTLT1', '_VEGLT1',
                            '_RFDRHV5',
                            'HLTHPLN1', 'MEDCOST',
                            'GENHLTH', 'MENTHLTH', 'PHYSHLTH', 'DIFFWALK',
                            'SEX', '_AGEG5YR', 'EDUCA', 'INCOME2']]

print(diabetes_df.head())
print(diabetes_df.shape)

# Drop rows with missing values

diabetes_df = diabetes_df.dropna()
print(diabetes_df.shape)
print(diabetes_df.head())

# Altering dataset to be more machine learning friendly.
# As a general rule 0's will be used to represent the absence of something or a no answer
# 1's will be used to represent the presence of something or a yes answer

# Editing DIABETE3 Column to be Binary
# Setting 1 for yes to Diabetes
# setting 0 to be for Pre diabetes or borderline, no diabetes or only during pregnancy
# all 7 (Don't Know) and 9 (refused to Answer) will be removed.

diabetes_df['DIABETE3'] = diabetes_df['DIABETE3'].replace({2: 0, 3: 0, 4: 0})
diabetes_df = diabetes_df[diabetes_df.DIABETE3 != 7]
diabetes_df = diabetes_df[diabetes_df.DIABETE3 != 9]
print(diabetes_df.DIABETE3.unique())

# Editing _RFHYPE5 - Adults who have been told they have high blood pressure by a health professional
# setting 1 to indicate High Blood Pressure and 0 to indicate no High Blood Pressure.
# all 9 (refused to Answer) will be removed.

diabetes_df['_RFHYPE5'] = diabetes_df['_RFHYPE5'].replace({1: 0, 2: 1})
diabetes_df = diabetes_df[diabetes_df._RFHYPE5 != 9]
print(diabetes_df._RFHYPE5.unique())

# Editing TOLDHI2 - Have you EVER been told by a health professional that your blood cholesterol is high?
# setting 0 to mean they have not been told they have high blood pressure and 1 to mean they have.
# all 7 (Don't Know) and 9 (refused to Answer) will be removed.

diabetes_df['TOLDHI2'] = diabetes_df['TOLDHI2'].replace({2: 0})
diabetes_df = diabetes_df[diabetes_df.TOLDHI2 != 7]
diabetes_df = diabetes_df[diabetes_df.TOLDHI2 != 9]
print(diabetes_df.TOLDHI2.unique())

# Editing _CHOLCHK - Cholesterol check within past five years
# setting 0 to mean they have not had their cholesterol checked in the last 5 years, and 1 to mean they have.
# all 9 (refused to Answer) will be removed.

diabetes_df['_CHOLCHK'] = diabetes_df['_CHOLCHK'].replace({3: 0, 2: 0})
diabetes_df = diabetes_df[diabetes_df._CHOLCHK != 9]
print(diabetes_df._CHOLCHK.unique())

# _BMI5 - Body Mass Index (BMI)
# needs to be divided by 100 for actual value.

diabetes_df['_BMI5'] = diabetes_df['_BMI5'].div(100).round(0)
print(diabetes_df['_BMI5'].unique())

# Editing SMOKE100 - Have you smoked at least 100 cigarettes in your entire life?
# setting 0 to mean they have are not smokers and 1 to mean they are.
# all 7 (Don't Know) and 9 (refused to Answer) will be removed.

diabetes_df['SMOKE100'] = diabetes_df['SMOKE100'].replace({2: 0})
diabetes_df = diabetes_df[diabetes_df.SMOKE100 != 7]
diabetes_df = diabetes_df[diabetes_df.SMOKE100 != 9]
print(diabetes_df.SMOKE100.unique())

# Editing CVDSTRK3 - (Ever told) you had a stroke.
# Setting 0 to mean they have not had a stroke and 1 meaning they have.
# all 7 (Don't Know) and 9 (refused to Answer) will be removed.

diabetes_df['CVDSTRK3'] = diabetes_df['CVDSTRK3'].replace({2: 0})
diabetes_df = diabetes_df[diabetes_df.CVDSTRK3 != 7]
diabetes_df = diabetes_df[diabetes_df.CVDSTRK3 != 9]
print(diabetes_df.CVDSTRK3.unique())

# Editing _MICHD - Respondents that have ever reported having coronary heart disease (CHD) or myocardial infarction (MI)
# setting 1 to mean patient reported having coronary heart disease (CHD) or myocardial infarction (MI)
# while 0 means they have not
diabetes_df['_MICHD'] = diabetes_df['_MICHD'].replace({2: 0})
print(diabetes_df._MICHD.unique())

# Editing _TOTINDA - Adults who reported doing physical activity or exercise during the past 30 days
# other than their regular job
# setting 0 to mean No physical activity in the last 30 days, 1 to mean had physical activity or exercise
# 9 (refused to Answer) will be removed.
diabetes_df['_TOTINDA'] = diabetes_df['_TOTINDA'].replace({2: 0})
diabetes_df = diabetes_df[diabetes_df._TOTINDA != 9]
print(diabetes_df._TOTINDA.unique())

# Editing _FRTLT1 - Consume Fruit 1 or more times per day
# Setting 0 to mean no fruit consumed per day.
# Setting 1 to mean they consumed 1 or more pieces of Fruit per day
# all responses of 9 (refused to Answer, Don't Know) will be removed.

diabetes_df['_FRTLT1'] = diabetes_df['_FRTLT1'].replace({2: 0})
diabetes_df = diabetes_df[diabetes_df._FRTLT1 != 9]
diabetes_df._FRTLT1.unique()

# Editing _VEGLT1 - Consume Vegetables 1 or more times per day
# Change 2 to 0. 0 for no vegetables consumed per day. 1 for consumes vegetables one or more times a day
# all 7 (Don't Know) and 9 (refused to Answer) will be removed.
diabetes_df['_VEGLT1'] = diabetes_df['_VEGLT1'].replace({2: 0})
diabetes_df = diabetes_df[diabetes_df._VEGLT1 != 9]
diabetes_df._VEGLT1.unique()

# Editing _RFDRHV5 - Heavy drinkers, adult men having more than 14 drinks per week
# adult women having more than 7 drinks per week
# changing 1 to 0 for no to heavy drinking and chnaging 2 to 1 for yes to heavy drinking
# all 7 (Don't Know) and 9 (refused to Answer) will be removed.
diabetes_df['_RFDRHV5'] = diabetes_df['_RFDRHV5'].replace({1: 0, 2: 1})
diabetes_df = diabetes_df[diabetes_df._RFDRHV5 != 9]
diabetes_df._RFDRHV5.unique()

# Editing HLTHPLN1 - Do you have any kind of health care coverage
# Changing 2 to 0 for No health care access, leaving 1 for yes to health care access
# all 7 (Don't Know) and 9 (refused to Answer) will be removed.
diabetes_df['HLTHPLN1'] = diabetes_df['HLTHPLN1'].replace({2: 0})
diabetes_df = diabetes_df[diabetes_df.HLTHPLN1 != 7]
diabetes_df = diabetes_df[diabetes_df.HLTHPLN1 != 9]
diabetes_df.HLTHPLN1.unique()

# Editing MEDCOST - Was there a time in the past 12 months when you needed to see a doctor
# but could not because of cost?
# Change 2 to 0 for no, 1 is already yes
# all 7 (Don't Know) and 9 (refused to Answer) will be removed.
diabetes_df['MEDCOST'] = diabetes_df['MEDCOST'].replace({2: 0})
diabetes_df = diabetes_df[diabetes_df.MEDCOST != 7]
diabetes_df = diabetes_df[diabetes_df.MEDCOST != 9]
diabetes_df.MEDCOST.unique()

# Editing GENHLTH - Health in General
# VAriable is ordinal - 1 - Excellent
# 2 - Very Good
# 3 - Good
# 4 - Fair
# 5 - Poor
# all 7 (Don't Know) and 9 (refused to Answer) will be removed.
diabetes_df = diabetes_df[diabetes_df.GENHLTH != 7]
diabetes_df = diabetes_df[diabetes_df.GENHLTH != 9]
diabetes_df.GENHLTH.unique()

# Editing MENTHLTH - for how many days during the past 30 days was your mental health not good?
# Number of days is in a scale of 0-30, Keeping this
# 88 to be changed to 0 for None (no bad mental health days)
# all 77 (Don't Know) and 99 (refused to Answer) will be removed.
diabetes_df['MENTHLTH'] = diabetes_df['MENTHLTH'].replace({88: 0})
diabetes_df = diabetes_df[diabetes_df.MENTHLTH != 77]
diabetes_df = diabetes_df[diabetes_df.MENTHLTH != 99]
diabetes_df.MENTHLTH.unique()

# Editing PHYSHLTH -  for how many days during the past days was your physical health not good?
# Number of days is in a scale of 0-30, Keeping this
# Setting 88 to 0 as it means no days of bad physical health
# all 77 (Don't Know) and 99 (refused to Answer) will be removed.
diabetes_df['PHYSHLTH'] = diabetes_df['PHYSHLTH'].replace({88: 0})
diabetes_df = diabetes_df[diabetes_df.PHYSHLTH != 77]
diabetes_df = diabetes_df[diabetes_df.PHYSHLTH != 99]
diabetes_df.PHYSHLTH.unique()

# Editing DIFFWALK - Do you have serious difficulty walking or climbing stairs?
# Setting 2 to 0 for No
# Keeping 1 is it is already yes
# all 7 (Don't Know) and 9 (refused to Answer) will be removed.
diabetes_df['DIFFWALK'] = diabetes_df['DIFFWALK'].replace({2: 0})
diabetes_df = diabetes_df[diabetes_df.DIFFWALK != 7]
diabetes_df = diabetes_df[diabetes_df.DIFFWALK != 9]
diabetes_df.DIFFWALK.unique()

# Editing SEX - Indicate sex of respondent.
# Setting 2 (Female) to 0 . Setting Male to 1
diabetes_df['SEX'] = diabetes_df['SEX'].replace({2: 0})
diabetes_df.SEX.unique()

# Editing _AGEG5YR - Fourteen-level age category
# Values of 1 - 13 represent age catagories starting at 18-24 and increasing in 5 year increments to 80+
# All responses of 14 are removed (don't know or missing)
diabetes_df = diabetes_df[diabetes_df._AGEG5YR != 14]
diabetes_df._AGEG5YR.unique()

# Editing EDUCA - Level of education completed
# Values of 1-6; Representing level of education gained by respondent
# all 9 (refused to Answer) will be removed.
diabetes_df = diabetes_df[diabetes_df.EDUCA != 9]
diabetes_df.EDUCA.unique()

# Editing INCOME2 - Is your annual household income from all sources
# Ordinal variable with values 1 - 8
# all 77 (Don't Know) and 99 (refused to Answer) will be removed.
diabetes_df = diabetes_df[diabetes_df.INCOME2 != 77]
diabetes_df = diabetes_df[diabetes_df.INCOME2 != 99]
diabetes_df.INCOME2.unique()

# Checking the shape of the dataframe (There is a class imbalance within the dataset)
print("Shape of the dataframe is:\n", diabetes_df.shape)

# Checking the number of Positive vs negative Diabetes Responses
print(diabetes_df.groupby(['DIABETE3']).size())

# Changing the column names to be more human readable
diabetes_df_clean = diabetes_df.rename(columns={'DIABETE3': 'Diabetes',
                                                '_RFHYPE5': "High_BP",
                                                'TOLDHI2': 'High_Cholesterol',
                                                '_CHOLCHK': 'Cholesterol_Check',
                                                '_BMI5': 'BMI',
                                                'SMOKE100': 'Ever_Smoked',
                                                'CVDSTRK3': 'Had_Stroke',
                                                '_MICHD': 'MI_or_CHD',
                                                '_TOTINDA': 'Physical_Activity',
                                                '_FRTLT1': 'Eats_Fruit',
                                                '_VEGLT1': 'Eats_Vegetables',
                                                '_RFDRHV5': 'Heavy_Drinker',
                                                'HLTHPLN1': 'Has_Health_Care',
                                                'MEDCOST': 'Couldnt_afford_doc',
                                                'GENHLTH': 'General_Health',
                                                'MENTHLTH': 'Mental_Health',
                                                'PHYSHLTH': 'Physical_Health',
                                                'DIFFWALK': 'Difficulty_Walking',
                                                'SEX': 'Sex',
                                                '_AGEG5YR': 'Age',
                                                'EDUCA': 'Education',
                                                'INCOME2': 'Income'
                                                })


# Checking that all column names have been correctly changed
print(diabetes_df_clean.columns.values)


# Saving the cleaned dataset to CSV file locally.
# diabetes_df_clean.to_csv('Data/BRFSS/Cleaned_Dataset.csv', sep=",", index=False)


# Removing the class imbalance in the data set
# first creating two dataframes, one with all diabetes and one with no diabetes
diabetes_pos = diabetes_df_clean['Diabetes'] == 1
pos_df = diabetes_df_clean[diabetes_pos]

diabetes_neg = diabetes_df_clean['Diabetes'] == 0
neg_df = diabetes_df_clean[diabetes_neg]

# Taking a random sample of the larger set of negative records
neg_balanced = neg_df.take(np.random.permutation(len(neg_df))[:35346])


# Appending both datasets back together to form final dataframe
diabetes_balanced = neg_balanced.append(pos_df, ignore_index=True)


print(diabetes_balanced.head())
print(diabetes_balanced.groupby(['Diabetes']).size())
print(diabetes_balanced.shape)

# Saving the balanced dataset to CSV file locally.
# diabetes_balanced.to_csv('Data/BRFSS/Balanced_Dataset.csv', sep=",", index=False)
