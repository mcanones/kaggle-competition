import numpy as np
from sklearn.preprocessing import MinMaxScaler

def cleanDiamonds(df, name):
    
    #STEP 1 > Rename columns
    df.rename(columns={'Unnamed: 0':'id'}, inplace=True)

    #STEP 2 > Convert categorical to numerical
    df.replace({'Ideal': 1, 'Premium': 2, 'Very Good':3, 'Good':4, 'Fair':5}, inplace=True)
    df.replace({'D':1, 'E':2, 'F':3, 'G':4, 'H':5, 'I':6, 'J':7}, inplace=True)
    df.replace({'IF':1, 'VVS1':2, 'VVS2':3, 'VS1':4, 'VS2':5, 'SI1':6, 'SI2':7, 'I1':8}, inplace=True)

    #STEP 3 > Outliers (IF NECESSARY) 
    #remove_outlier(df,'table')

    #STEP 4 > Remove very correlated features 
    df.drop(columns=['x','y','z'], inplace=True)

    #STEP 5 > Scaling features
    df[['table','carat','depth']] = MinMaxScaler().fit_transform(df[['table','carat','depth']])
    
    #Saving
    df.to_csv(f'./output/{name}.csv', header = True, index=False)  

    return df

def remove_outlier(df, col):

    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_clean = df.loc[(df[col] > fence_low) & (df[col] < fence_high)]

    return df_clean