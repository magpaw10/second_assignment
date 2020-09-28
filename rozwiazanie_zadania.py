import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

survey_results_public_file_path = 'survey_results_public.csv'
survey_results_public_data = pd.read_csv(survey_results_public_file_path,
                            usecols=['Respondent','Age','WorkWeekHrs','ConvertedComp','CodeRevHrs','YearsCode','Hobbyist','OpSys'],
                            index_col='Respondent')
#print (survey_results_public_data.dtypes)
survey_results_public_data.dropna(inplace=True)
survey_results_public_data.replace(to_replace={'Less than 1 year': '0',
                              'More than 50 years': '51'},
                             inplace=True)
survey_results_public_data = survey_results_public_data.astype({'YearsCode': 'int64'},copy=False)
#print (survey_results_public_data.dtypes)
survey_results_public_data.replace(to_replace={'Yes': 1,
                              'No': 0 },
                             inplace=True)
#print (survey_results_public_data.corr())
one_hot = pd.get_dummies(survey_results_public_data['OpSys'])
#print (one_hot)
survey_results_public_data = survey_results_public_data.drop('OpSys', axis = 1)
survey_results_public_data = survey_results_public_data.join(one_hot)
'''print (survey_results_public_data)'''
#correlated
# plt.plot(survey_results_public_data['Age'], survey_results_public_data['YearsCode'], 'ro', markersize=0.1)
# plt.title('Coding experince in accorance to the age')
# plt.xlabel('Age')
# plt.ylabel('Years of Code')
# plt.show()
#uncorrelated
# survey_results_public_data = survey_results_public_data[(survey_results_public_data['WorkWeekHrs'] <=84)]
# plt.plot(survey_results_public_data['WorkWeekHrs'], survey_results_public_data['YearsCode'], 'ro', markersize=0.1)
# plt.title('Weekly working hours in accordance to experince')
# plt.xlabel('Hours per week')
# plt.ylabel('Years of Code')
# plt.show()

# PART 3
# quantil method
q1 = survey_results_public_data.quantile(0.25)
q3 = survey_results_public_data.quantile(0.75)
iqr = q3 - q1
print (iqr)

# buiding a summary table including numerical values of outliers (quantil method)
low_boundary = (q1 - 1.5 * iqr)
upp_boundary = (q3 + 1.5 * iqr)
num_of_outliers_L = (survey_results_public_data[iqr.index] < low_boundary).sum()
num_of_outliers_U = (survey_results_public_data [iqr.index] > upp_boundary).sum()
outliers_15iqr = pd.DataFrame({'lower_boundary':low_boundary, 'upper_boundary':upp_boundary,'num_of_outliers_L':num_of_outliers_L, 'num_of_outliers_U':num_of_outliers_U})
print (outliers_15iqr)

# removing outliers (quantil method)
data_without_outliers = survey_results_public_data.copy()
print (data_without_outliers.shape[0]) #number of data before removing outliers
for row in outliers_15iqr.iterrows():
    data_without_outliers = data_without_outliers[(data_without_outliers[row[0]] >= row[1]['lower_boundary']) & (data_without_outliers[row[0]] <= row[1]['upper_boundary'])]
print (data_without_outliers.shape[0]) #number of data after outliners removal  


# standard deviation method
elements = ['WorkWeekHrs','CodeRevHrs']
sigma = survey_results_public_data[elements].std()
average = survey_results_public_data[elements].mean()

print(sigma)
print (average)

# buiding a summary table including numerical values of outliers (standard deviation)

low_boundary = (average - 3 * sigma)
upp_boundary = (average + 3 * sigma)
num_of_outliers_L = (survey_results_public_data[elements] < low_boundary).sum()
num_of_outliers_U = (survey_results_public_data[elements] > upp_boundary).sum()
outliers_3sigma = pd.DataFrame({'lower_boundary':low_boundary, 'upper_boundary':upp_boundary,'num_of_outliers_L':num_of_outliers_L, 'num_of_outliers_U':num_of_outliers_U})
print (outliers_3sigma)

#removing outliers (standard deriviation)
data_without_outliers_sd = survey_results_public_data.copy()
print (data_without_outliers_sd.shape[0]) #number of data before removing outliers
for row in outliers_3sigma.iterrows():
    data_without_outliers_sd = data_without_outliers_sd[(data_without_outliers_sd[row[0]] >= row[1]['lower_boundary']) & (data_without_outliers_sd[row[0]] <= row[1]['upper_boundary'])]
print (data_without_outliers_sd.shape[0]) #number of data after outliners removal  

# PART 4
# charts

# box plot visualisation
sns.boxplot(y=data_without_outliers['WorkWeekHrs'], x=data_without_outliers['CodeRevHrs'], data=data_without_outliers)
plt.show()

# linear regression visualiosation 
sns.regplot(y=data_without_outliers['WorkWeekHrs'], x=data_without_outliers['Age'])
plt.show()

# joint plot method
sns.jointplot(y=data_without_outliers['WorkWeekHrs'], x=data_without_outliers['Age'], data=data_without_outliers, kind='hex')
plt.show()

print (data_without_outliers)

# linear regression => MODELS
# for 1 x
reg = linear_model.LinearRegression()
reg.fit(data_without_outliers[['WorkWeekHrs']], data_without_outliers['CodeRevHrs'])
print (reg.predict([[12]]))
print (reg.predict([[40]]))
mse = np.mean((reg.predict(data_without_outliers[['WorkWeekHrs']]) - data_without_outliers[['CodeRevHrs']])**2)
print("Error: ", mse)

# for 2 x
reg = linear_model.LinearRegression()
reg.fit(data_without_outliers[['WorkWeekHrs','YearsCode']], data_without_outliers[['CodeRevHrs']])
print (reg.coef_)

# 
reg = linear_model.LinearRegression()
reg.fit(data_without_outliers[['WorkWeekHrs','YearsCode','Age']], data_without_outliers[['CodeRevHrs']])
