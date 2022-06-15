import sys
import pandas as pd
import path
import os

def remove_space():
    """ Trim spaces and save file back as a csv."""
    df = pd.read_csv("./data/census.csv", skipinitialspace=True)   
    new_col_names = [] 
    list = []
    for i in df.columns :
        j = i.strip()
        if df[i].dtype == object:
            list.append(i)
            df[i].str.strip()
        new_col_names.append(j)
    df.columns = new_col_names
    df.to_csv("./data/cleaned_census.csv",index=False)

if __name__ == "__main__":
    remove_space()