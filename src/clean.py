import numpy as np

def cleanDiamonds(df, name):
    
    #STEP 1 > Rename columns
    df.rename(columns={'Unnamed: 0':'id'}, inplace=True)

    #STEP 2 > Convert categorical to numerical
    df.replace({'Ideal': 5, 'Premium': 4, 'Very Good':3, 'Good':2, 'Fair':1}, inplace=True)
    df.replace({'D':7, 'E':6, 'F':5, 'G':4, 'H':3, 'I':2, 'J':1}, inplace=True)
    df.replace({'IF':8, 'VVS1':7, 'VVS2':6, 'VS1':5, 'VS2':4, 'SI1':3, 'SI2':2, 'I1':1}, inplace=True)

    #STEP 3 > Remove very correlated features 
    df.drop(columns=['x','y','z'], inplace=True)

    #Saving
    df.to_csv(f'./output/{name}.csv', header = True, index=False)  

    return df

"""
#STEP > Outliers (IF NECESSARY) 
#remove_outlier(df,'table')

def remove_outlier(df, col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_clean = df.loc[(df[col] > fence_low) & (df[col] < fence_high)]
    return df_clean
"""