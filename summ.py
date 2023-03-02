import numpy as np
import glob
import pdb
import argparse

parser = argparse.ArgumentParser(description='Exp config')

parser.add_argument('--n', type=str, default=1)
parser.add_argument('--method', type=str, default='ogd')
parser.add_argument('--b', type=str, default='s_minus')
args = parser.parse_args()

exps = glob.glob('results/*{}_*.txt'.format(args.b))
exps = sorted(exps)
for exp in exps:
    with open(exp) as f:
        data = f.readlines()
    s = exp.split('/')[1].split('2022')[0][:-1]
    arr = s.split('_')
    method = '_'.join(arr[:-2])
    header = 'Method: {}'.format(method)
    buff = header+'\n' +  data[-4] + data[-3] + data[-2] + data[-1]
    print(buff)
    
    
