import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

survey_results_public_file_path = 'survey_results_public.csv'
survey_results_public_data = pd.read_csv(survey_results_public_file_path,
                            usecols=['Respondent','Age','WorkWeekHrs','ConvertedComp','CodeRevHrs','YearsCode'],
                            index_col='Respondent')
#print (survey_results_public_data.dtypes)
survey_results_public_data.dropna(inplace=True)
survey_results_public_data.replace(to_replace={'Less than 1 year': '0',
                              'More than 50 years': '51'},
                             inplace=True)
survey_results_public_data = survey_results_public_data.astype({'YearsCode': 'int64'},copy=False)
print (survey_results_public_data.dtypes)
print (survey_results_public_data.corr())

#correlated
plt.plot(survey_results_public_data['Age'], survey_results_public_data['YearsCode'], 'ro', markersize=0.1)
plt.title('Coding experince in accorance to the age')
plt.xlabel('Age')
plt.ylabel('Years of Code')
plt.show()
#uncorrelated
survey_results_public_data = survey_results_public_data[(survey_results_public_data['WorkWeekHrs'] <=84)]
plt.plot(survey_results_public_data['WorkWeekHrs'], survey_results_public_data['YearsCode'], 'ro', markersize=0.1)
plt.title('Weekly working hours in accordance to experince')
plt.xlabel('Hours per week')
plt.ylabel('Years of Code')
plt.show()