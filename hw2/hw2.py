import sys
import csv

def getTable(file):
    with open(file, 'r') as csvFile:
        csvObj = csv.reader(csvFile)
        table = []
        for row in csvObj:
            table.append(row)
    return table

def main():
    print(getTable(sys.argv[1]))

if __name__ == '__main__':
    main()