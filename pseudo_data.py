import os
import numpy as np

with open('./pseudo_f', 'w') as f:
    for i in xrange(3000):
        f.write('\t'.join([str(i)]+map(str,(np.random.rand(4000))))+'\n')


with open('./pseudo_s', 'w') as f:
    for i in xrange(1000):
        cid=np.random.randint(3000)
        did=np.random.randint(800)
        score=np.random.random()
        f.write('\t'.join(map(str, [cid, did, score]))+'\n')
