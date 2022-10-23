import csv
import pandas as pd

def read_csv(filename, names, delimiter='t'):
    if delimiter == 't':
        sep = '\t'
    elif delimiter == 'b':
        sep = ' '
    else:
        sep = delimiter
    return pd.read_csv(filename, sep=sep, quoting=csv.QUOTE_NONE, skip_blank_lines=False, header=None, names=names)
