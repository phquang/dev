from continuum import ContinualScenario
from continuum.datasets import CTRLminus, CTRLplus, CTRLplastic, CTRLin, CTRLout
import numpy as np
import pdb


path = './temp'
download = True
d = CTRLminus
s_train = ContinualScenario(d(path, split="train", download=download))
s_val = ContinualScenario(d(path, split="val", download=download))
s_test = ContinualScenario(d(path, split="test", download=download))

print("Amount of tasks", len(s_train), len(s_val), len(s_test))
'''
for i, (tr_set, va_set, te_set) in enumerate(zip(s_train, s_val, s_test)):
    assert np.unique(tr_set._y).tolist() == np.unique(va_set._y).tolist() == np.unique(te_set._y).tolist()
    print(f"Task {i}, classes: {np.unique(tr_set._y)}, train={len(tr_set)}, val={len(va_set)}, test={len(te_set)}")
    print("---------------")
'''

from torch.utils.data import DataLoader

s_tr = DataLoader(s_train[1], batch_size=64, shuffle=True)

for x,y,t in s_tr:
    pdb.set_trace()
