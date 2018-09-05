import pandas as pd
import numpy as np

widths = [1,1,1,1]
heights = [1,1,1,3]

def header():
    print ("\n")
    print ("-"*50)
    print ("\n")

filename = 'weather.txt'

df = pd.read_csv(filename)
header()


print(df.dtypes)

print(df.describe())
header()
print(df.sort_values('record_high', ascending=False))