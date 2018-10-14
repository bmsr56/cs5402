import sys
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd

def getDF(csvFile):
    table = pd.read_csv(csvFile, sep= ',')
    return table

def main():
    df = getDF(sys.arg[1])


    return

if __name__ == '__main__':
    main()