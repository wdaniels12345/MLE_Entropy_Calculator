#Walker Daniels Robert Taylor Oscar Salinas Zachary Urrutia 
#Human Computer Interaction 
#CSV Project 
#11/1/22

import csv
import math
import pandas as pd
import numpy as np
import collections
import sys
from collections import defaultdict
from math import e
import scipy.stats as ss
import matplotlib.pyplot as plt


#This section of the code accomplishes the task of deleting all rows in the Human.csv file that have the ? character 
#It reads the provided CSV file and writes to another one title Human_edit.csv for every row that does not start with ?
with open('Human.csv', 'r') as inp, open('Human_edit.csv', 'w') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if row[1] != " ?" and row[6] != " ?" and row[13] != " ?":
            writer.writerow(row)
#This portion of the code is specifically meant for the writers so as to easily verify what column you are currently working with in the csv file 
with open('Human_edit.csv', 'r') as infile:
    reader = csv.reader(infile, delimiter=",")
    header = next(reader)
    for row in reader:
        age = row[0]
        employment = row[1]
        idnNumber = row[2]
        education = row[3]
        educationRank = row[4]
        marriageStatus = row[5]
        speciality = row[6]
        filingStatus = row[7]
        ethnicity = row[8]
        sex = row[9]
        numberOne = row[10]
        numberTwo = row[11]
        numberThree = row[12]
        country = row[13]
        salary = row[14]



#This is where the occurences for each column are switched to arbitrary values 
df = pd.read_csv("Human_edit.csv")
d = {'Private': 0, 'Self-emp-not-inc': 1, 'Local-gov': 2, 'Self-emp-inc': 3, 'Federal-gov': 4, 'Without-pay': 5, 'Never-worked': 6}
df['x2'] = df['x2'].map(d)
d = {'HS-grad': 0, 'Some-college': 1, 'Bachelors': 2, 'Masters': 3, 'Assoc-voc': 4, '11th': 5, 'Assoc-acdm': 6, '10th': 7, '7th-8th': 8, 'Prof-school': 9, '9th': 10, '12th': 11, 'Doctorate': 12, '5th-6th': 13, '1st-4th': 14, 'Preschool': 15}
df['x4'] = df['x4'].map(d)
d = {'Married-civ-spouse': 0, 'Never-married': 1, 'Divorced': 2, 'Separated': 3, 'Widowed': 4, 'Married-spouse-absent': 5, 'Married-AF-spouse': 6}
df['x6'] = df['x6'].map(d)
d = {'Craft-repair': 0, 'Exec-managerial': 1, 'Prof-specialty': 2, 'Adm-clerical': 3, 'Sales': 4, 'Other-service': 5, 'Machine-op-inspct': 6, 'Transport-moving': 7, 'Handlers-cleaners': 8, 'Tech-support': 9, 'Farming-fishing': 10, 'Protective-serv': 11, 'Priv-house-serv': 12, 'Armed-Forces': 14}
df['x7'] = df['x7'].map(d)
d = {'Husband': 0, 'Not-in-family': 1, 'Own-child': 2, 'Unmarried': 3, 'Wife': 4, 'Other-relative': 5}
df['x8'] = df['x8'].map(d)
d = {'White': 0, 'Black': 1, 'Asian-Pac-Islander': 2, 'Amer-Indian-Eskimo': 3, 'Other': 4}
df['x9'] = df['x9'].map(d)
d = {'Male': 0, 'Female': 1}
df['x10'] = df['x10'].map(d)
d = {'United-States': 0, 'Mexico': 1, 'Phillipines': 2, 'Puerto-Rico': 3, 'Germany': 4, 'El-Salvador': 5, 'Canada': 6, 'India': 7, 'China': 8, 'Italy': 9, 'Cuba': 10, 'Japan': 11, 'Portugal': 12, 'South': 14, 'England': 15, 'Haiti': 16, 'Dominican-Republic': 17, 'Columbia': 18, 'Poland': 19, 'Jamaica': 20, 'Vietnam': 21, 'Ecuador': 22, 'Nicaruaga': 23, 'Greece': 24, 'Ireland': 25, 'Guatemala': 26, 'Taiwan': 27, 'Thailand': 28, 'Peru': 29, 'France': 30, 'Honduras': 31, 'Iran': 32, 'Scotland': 33, 'Outlying-US(Guam-USVI-etc)': 34, 'Cambodia': 35, 'Hong': 36, 'Lungary': 37, 'Laos': 38, 'Trinadad&Tobago': 39, 'Yugoslavia': 40}
df['x14'] = df['x14'].map(d)
d = {'<=50K.': 0, '>50K.': 1}
df['Y'] = df['Y'].map(d)


        
#Here is where the edited file is split into two seperate files of seventy and thirty percent using pandas
df = pd.read_csv('Human_edit.csv')
df['split'] = np.random.randn(df.shape[0], 1)

msk = np.random.rand(len(df)) <= 0.7

seventy = df[msk]
thirty = df[~msk]

#And here is where the percentages are written into two new files 
seventy.to_csv('seventy.csv', index=False)
thirty.to_csv('thirty.csv', index=False)

#Here we open the CSV file holding seventy percent of the total and display each occurence and how often they occur in a particular row
df = pd.read_csv('seventy.csv', index_col=0)




print()
print()
print("----------------------------------------")
print("Emplpoyment Occurences: ")
print("----------------------------------------")



#This is the block of code that displays each occurence based on the column we choose to use... It is written 15 times for the 15 columns in the CSV 
df['x2'].value_counts()
employment = df['x2'].value_counts()
print(employment)

df['x3'].value_counts()
print()
print("Employment Number Occurences")
employmentNum= df['x3'].value_counts()
print(employmentNum)

df['x4'].value_counts()
print()
print("Employment Education Occurences")
employmentEdu = df['x4'].value_counts()
print(employmentEdu)

df['x5'].value_counts()
print()
print("Employment Education Rank Occurences")
employmentEduRank = df['x5'].value_counts()
print(employmentEduRank)

df['x6'].value_counts()
print()
print("Employment Marriage Occurences")
employmentMar = df['x6'].value_counts()
print(employmentMar)

df['x7'].value_counts()
print()
print("Employment Specialty Occurences")
employmentSpec = df['x7'].value_counts()
print(employmentSpec)

df['x8'].value_counts()
print()
print("Employment Filing Occurences")
employmentFil = df['x8'].value_counts()
print(employmentFil)

df['x9'].value_counts()
print()
print("Employment Race Occurences")
employmentRac = df['x9'].value_counts()
print(employmentRac)

df['x10'].value_counts()
print()
print("Employment Sex Occurences")
employmentSex = df['x10'].value_counts()
print(employmentSex)

df['x11'].value_counts()
print()
print("Employment number 1 Occurences")
employmentNumOne = df['x11'].value_counts()
print(employmentNumOne)

df['x12'].value_counts()
print()
print("Employment number 2 Occurences")
employmentNumTwo = df['x12'].value_counts()
print(employmentNumTwo)

df['x13'].value_counts()
print()
print("Employment number 3 Occurences")
employmentNumTre = df['x13'].value_counts()
print(employmentNumTre)

df['x14'].value_counts()
print()
print("Employment Country Occurences")
employmentCountry = df['x14'].value_counts()
print(employmentCountry)

df['Y'].value_counts()
print()
print("Employment Salary Occurences")
employmentSalary = df['Y'].value_counts()
print(employmentSalary)





###########################################
#MLE CALCULATION
###########################################



numOfLinesMoreThan50k=0
numOfLinesLessThan50k=0
numOfLinesWithBothM=0
numOfLinesWithBothF=0
numOfLinesWithBothMale=0
numOfLinesWithBothFemale=0

#This is the logic used to calculate our a and b for our MLE calculation using counts based on some if statements
with open('Seventy.csv', 'r') as infile:
    reader = csv.reader(infile, delimiter=",")
    header = next(reader)
    for row in reader:

        if row[14] != " >50K." :
            numOfLinesLessThan50k+=1
            if row[9] != " Male":
                numOfLinesWithBothF+=1
            if row[9] != " Female":
                numOfLinesWithBothM+=1

        if row[14] !=  " <=50K.":
            numOfLinesMoreThan50k+=1
            if row[9] != " Male":
                numOfLinesWithBothFemale+=1
            if row[9] != " Female":
                numOfLinesWithBothMale+=1
#Here is where mle is calculated for male and female accordingly 
probabilityofAnAttributeSexMale=(numOfLinesWithBothM/numOfLinesLessThan50k)

probabilityofAnAttributeSexFemale=(numOfLinesWithBothF/numOfLinesLessThan50k)
print()
print()
print("----------------------------------------")
print("70% MLE Calucation of An attribute Gender: ")
print("----------------------------------------")

print()
print("Probability of Sex Male and >50k: ")
print(probabilityofAnAttributeSexMale)
print("Probability of Sex Female and >50k: ")
print(probabilityofAnAttributeSexFemale)
print("Number of Rows with >50k: ")
print(numOfLinesLessThan50k)
print("Number of rows with Sex Male and >50k: ")
print(numOfLinesWithBothM)
print("Number of rows with Sex Female and >50k: ")
print(numOfLinesWithBothF)
print()
print()
#resetting the probability variables
probabilityofAnAttributeSexMale1=0
probabilityofAnAttributeSexFemale1=0

probabilityofAnAttributeSexMale1=(numOfLinesWithBothMale/numOfLinesMoreThan50k)

probabilityofAnAttributeSexFemale1=(numOfLinesWithBothFemale/numOfLinesMoreThan50k)

print("Probability of Sex Male and <=50k: " )
print(probabilityofAnAttributeSexMale1)
print("Probability of Sex Female and <=50k: ")
print(probabilityofAnAttributeSexFemale1)
print("Number of Rows with <=50k: ")
print(numOfLinesMoreThan50k)
print("Number of rows with Sex Male and <=50k: ")
print(numOfLinesWithBothMale)
print("Number of rows with Sex Female and <=50k: ")
print(numOfLinesWithBothFemale)


# 100% ENTROPY CALCULATION
###########################################

df = pd.read_csv('Human.csv', index_col=0)

#This is our entropy function using pandas
def pandas_entropy(column, base=None):
    vc = pd.Series(column).value_counts(normalize=True, sort=False)
    base = e if base is None else base
    return -(vc * np.log(vc)/np.log(base)).sum()

print()
print()
print("----------------------------------------")
print("100% Entropy Calculations:")
print("----------------------------------------")

#This is where entropy for each column of our file is calculated

print()
pandas_entropy(df['x2'])
print()
print("Employment Occurences entropy")
employmentOccEnt = pandas_entropy(df['x2'])
print(employmentOccEnt)

pandas_entropy(df['x3'])
print()
print("Employment Number Entropy")
employmentNumEnt = pandas_entropy(df['x3'])
print(employmentNumEnt)

pandas_entropy(df['x4'])
print()
print("Employment Education Entropy")
employmentEduEnt = pandas_entropy(df['x4'])
print(employmentEduEnt)

pandas_entropy(df['x5'])
print()
print("Employment Education Rank Entropy")
employmentEdrEnt = pandas_entropy(df['x5'])
print(employmentEdrEnt)

pandas_entropy(df['x6'])
print()
print("Employment Marriage Entropy")
employmentMarEnt = pandas_entropy(df['x6'])
print(employmentMarEnt)

pandas_entropy(df['x7'])
print()
print("Employment Specialty Entropy")
employmentSpcEnt = pandas_entropy(df['x7'])
print(employmentSpcEnt)

pandas_entropy(df['x8'])
print()
print("Employment Filing Entropy")
employmentFilEnt = pandas_entropy(df['x8'])
print(employmentFilEnt)

pandas_entropy(df['x9'])
print()
print("Employment Race Entropy")
employmentRacEnt = pandas_entropy(df['x9'])
print(employmentRacEnt)

pandas_entropy(df['x10'])
print()
print("Employment Sex Entropy")
employmentSexEnt = pandas_entropy(df['x10'])
print(employmentSexEnt)

pandas_entropy(df['x11'])
print()
print("Employment Number 1 Entropy")
employmentNmOEnt = pandas_entropy(df['x11'])
print(employmentNmOEnt)

pandas_entropy(df['x12'])
print()
print("Employment Number 2 Entropy")
employmentNmTEnt = pandas_entropy(df['x12'])
print(employmentNmTEnt)

pandas_entropy(df['x13'])
print()
print("Employment Number 3 Entropy")
employmentNmREnt = pandas_entropy(df['x13'])
print(employmentNmREnt)

pandas_entropy(df['x14'])
print()
print("Employment Country Entropy")
employmentCouEnt = pandas_entropy(df['x14'])
print(employmentCouEnt)

pandas_entropy(df['Y'])
print()
print("Employment Salary Entropy")
employmentSalEnt = pandas_entropy(df['Y'])
print(employmentSalEnt)

print()



#Here Entropy of employees race is calculated for all employees with <=50k in salary and >50k in salary. They are then compared to determine which is higher
with open('Human.csv', 'r') as inp, open('salaryEnt1.csv', 'w') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if row[14] != " <=50K.":
            writer.writerow(row)

with open('Human.csv', 'r') as inp, open('salaryEnt2.csv', 'w') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if row[14] != " >50K.":
            writer.writerow(row)

df = pd.read_csv('SalaryEnt1.csv', index_col=0)

pandas_entropy(df['x9'])
print()
print("Employment Race Entropy of employees who make <=50k annually.")
employmentRacEntOne = pandas_entropy(df['x9'])
print(employmentRacEntOne)

df = pd.read_csv('SalaryEnt2.csv', index_col=0)

pandas_entropy(df['x9'])
print()
print("Employment Race Entropy of employees who make >50k annually.")
employmentRacEntTwo = pandas_entropy(df['x9'])
print(employmentRacEntTwo)
print()

if employmentRacEntOne > employmentRacEntTwo:
    print("The portion of employees who make <=50k has a higher entropy than the portion of employees that makes >50k in regards to the race of employees.")

if employmentRacEntOne < employmentRacEntTwo:
    print("The portion of employees who make <=50k has a lower entropy than the portion of employees that makes >50k in regards to the race of employees.")

###########################################
#30% ENTROPY CALCULATION
###########################################

df = pd.read_csv('Thirty.csv', index_col=0)

#This is our entropy function using pandas
def pandas_entropy(column, base=None):
    vc = pd.Series(column).value_counts(normalize=True, sort=False)
    base = e if base is None else base
    return -(vc * np.log(vc)/np.log(base)).sum()

print()
print()
print("----------------------------------------")
print("30% Entropy Calculations:")
print("----------------------------------------")

#This is where entropy for each column of our file is calculated

print()
pandas_entropy(df['x2'])
print()
print("Employment Occurences entropy")
employmentOccEntThirty = pandas_entropy(df['x2'])
print(employmentOccEntThirty)

pandas_entropy(df['x3'])
print()
print("Employment Number Entropy")
employmentNumEntThirty = pandas_entropy(df['x3'])
print(employmentNumEntThirty)

pandas_entropy(df['x4'])
print()
print("Employment Education Entropy")
employmentEduEntThirty = pandas_entropy(df['x4'])
print(employmentEduEntThirty)

pandas_entropy(df['x5'])
print()
print("Employment Education Rank Entropy")
employmentEdrEntThirty = pandas_entropy(df['x5'])
print(employmentEdrEntThirty)

pandas_entropy(df['x6'])
print()
print("Employment Marriage Entropy")
employmentMarEntThirty = pandas_entropy(df['x6'])
print(employmentMarEntThirty)

pandas_entropy(df['x7'])
print()
print("Employment Specialty Entropy")
employmentSpcEntThirty = pandas_entropy(df['x7'])
print(employmentSpcEntThirty)

pandas_entropy(df['x8'])
print()
print("Employment Filing Entropy")
employmentFilEntThirty = pandas_entropy(df['x8'])
print(employmentFilEntThirty)

pandas_entropy(df['x9'])
print()
print("Employment Race Entropy")
employmentRacEntThirty = pandas_entropy(df['x9'])
print(employmentRacEntThirty)

pandas_entropy(df['x10'])
print()
print("Employment Sex Entropy")
employmentSexEntThirty = pandas_entropy(df['x10'])
print(employmentSexEntThirty)

pandas_entropy(df['x11'])
print()
print("Employment Number 1 Entropy")
employmentNmOEntThirty = pandas_entropy(df['x11'])
print(employmentNmOEntThirty)

pandas_entropy(df['x12'])
print()
print("Employment Number 2 Entropy")
employmentNmTEntThirty = pandas_entropy(df['x12'])
print(employmentNmTEntThirty)

pandas_entropy(df['x13'])
print()
print("Employment Number 3 Entropy")
employmentNmREntThirty = pandas_entropy(df['x13'])
print(employmentNmREntThirty)

pandas_entropy(df['x14'])
print()
print("Employment Country Entropy")
employmentCouEntThirty = pandas_entropy(df['x14'])
print(employmentCouEntThirty)

pandas_entropy(df['Y'])
print()
print("Employment Salary Entropy")
employmentSalEntThirty = pandas_entropy(df['Y'])
print(employmentSalEntThirty)

#Here Entropy of employees race is calculated for all employees with <=50k in salary and >50k in salary. They are then compared to determine which is higher
with open('Thirty.csv', 'r') as inp, open('salaryEnt1.csv', 'w') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if row[14] != " <=50K.":
            writer.writerow(row)

with open('Thirty.csv', 'r') as inp, open('salaryEnt2.csv', 'w') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if row[14] != " >50K.":
            writer.writerow(row)
    

df = pd.read_csv('SalaryEnt1.csv', index_col=0)

pandas_entropy(df['x9'])
print()
print("Employment Race Entropy of employees who make <=50k annually.")
employmentRacEntOne = pandas_entropy(df['x9'])
print(employmentRacEntOne)

df = pd.read_csv('SalaryEnt2.csv', index_col=0)

pandas_entropy(df['x9'])
print()
print("Employment Race Entropy of employees who make >50k annually.")
employmentRacEntTwo = pandas_entropy(df['x9'])
print(employmentRacEntTwo)
print()

if employmentRacEntOne > employmentRacEntTwo:
    print("The portion of employees who make <=50k has a higher entropy than the portion of employees that makes >50k in regards to the race of employees.")

if employmentRacEntOne < employmentRacEntTwo:
    print("The portion of employees who make <=50k has a lower entropy than the portion of employees that makes >50k in regards to the race of employees.")

print()

print()
print("----------------------------------------")
print("Entropy Accuracy")
print("----------------------------------------")
print()

print("Employment Occurences Accuracy")
print(employmentOccEntThirty/employmentOccEnt)

print("Employment Number Accuracy")
print(employmentNumEntThirty/employmentNumEnt)

print("Employment Education Accuracy")
print(employmentEduEntThirty/employmentEduEnt)

print("Employment Education Rank Accuracy")
print(employmentEdrEntThirty/employmentEdrEnt)

print("Employment Marriage Accuracy")
print(employmentMarEntThirty/employmentMarEnt)

print("Employment Specialty Accuracy")
print(employmentSpcEntThirty/employmentSpcEnt)

print("Employment Filing Accuracy")
print(employmentFilEntThirty/employmentFilEnt)

print("Employment Race Accuracy")
print(employmentRacEntThirty/employmentRacEnt)

print("Employment Sex Accuracy")
print(employmentSexEntThirty/employmentSexEnt)

print("Employment Number 1 Accuracy")
print(employmentNmOEntThirty/employmentNmOEnt)

print("Employment Number 2 Accuracy")
print(employmentNmTEntThirty/employmentNmTEnt)

print("Employment Number 3 Accuracy")
print(employmentNmREntThirty/employmentNmREnt)

print("Employment Country Accuracy")
print(employmentCouEntThirty/employmentCouEnt)

print("Employment Salary Accuracy")


            






