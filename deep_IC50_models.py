import numpy as np
import mxnet as mx
import random
import sys, collections
import logging
class Batch:
    def __init__(self, data_names, data, label_names, label, pad=0):
        self.data=data
        self.data_names=data_names
        self.label=label
        self.label_names=label_names

    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    def provide_label(self):
        return [(n,x.shape) for n, x in zip(self.label_names, self.label)]


class DataIter(mx.io.DataIter):
    def __init__(self, data, cid2feature, num_featuers,  batch_size):
        super(DataIter, self).__init__()
        self.batch_size=batch_size
        self.cid2feature = cid2feature
        self.data=data
        self.num_features = num_features
        self.provide_data=[('cell',(self.batch_size, self.num_features)), ('drug', (self.batch_size,))]
        self.provide_label = [('score', (self.batch_size, ))]

    def __iter__(self):
        for k in xrange(len(self.data)/self.batch_size):
            cells=[]
            drugs=[]
            scores=[]
            #Generate each batch data and yield the result
            for i in xrange(self.batch_size):
                j=k*self.batch_size+i
                cid, did, score=self.data[j]
                cells.append(self.cid2feature[cid])
                drugs.append(did)
                scores.append(score)
            data_all=[mx.nd.array(cells), mx.nd.array(drugs)]
            label_all=[mx.nd.array(scores)]
            data_names=['cell', 'drug']
            label_names=['score']

            data_batch=Batch(data_names, data_all, label_names, label_all)
            yield data_batch

    def reset(self):
        random.shuffle(self.data)

def RMSE(label, pred):
    ret=0.0
    n=0.0
    pred=pred.flatten()
    for i in xrange(len(label)):
        ret+=(label[i]-pred[i])**2
        n+=1.0
    return np.sqrt(ret/n)

def get_data(feedback_data, cid2feature, num_features, batch_size, validation=0.1):
    if validation:
        valid_ins = np.random.choice(feedback_data.shape[0], int(feedback_data.shape[0]*validation))
    train_ins = list(set(range(feedback_data.shape[0]))-set(valid_ins))

    return (DataIter(feedback_data[train_ins], cid2feature, num_features, batch_size), DataIter(feedback_data[valid_ins], cid2feature, num_features, batch_size))

def train(feedback_data, cid2feature, network, scoreFile, num_hidden, num_features, batch_size, num_epoch, learning_rate):
    train, test= get_data(feedback_data, cid2feature, num_features, batch_size)
    model=mx.mod.Module(network, data_names=['cell', 'drug'], label_names=['score'], context=[mx.gpu()])
    model.bind(data_shapes=train.provide_data, label_shapes=train.provide_label)
    model.init_params(initializer=mx.init.Uniform(scale=.1))
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)
    fileName = scoreFile+'-'+'-'.join(map(str, num_hidden))+'-'+str(batch_size)+'-'+str(num_epoch)+'-'+str(learning_rate)
    logPath='./results'
    fileHandler = logging.FileHandler("{0}/{1}.log".format(logPath, fileName))
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    rootLogger.addHandler(consoleHandler)


    model.init_optimizer(optimizer='adam', optimizer_params={'learning_rate':learning_rate})
    model.fit(train,eval_data=test, num_epoch=num_epoch, eval_metric='rmse', batch_end_callback=mx.callback.Speedometer(500) )

def build_network(num_features, num_drug, num_hidden):
    cell=mx.sym.Variable('cell')
    drug=mx.sym.Variable('drug')
    score = mx.sym.Variable('score')

    #Drug part
    drug=mx.sym.Embedding(data=drug, input_dim = num_drug, output_dim = num_hidden[-1])
    drug=mx.sym.Dropout(drug, p=0.2)
    drug=mx.sym.FullyConnected(data=drug, name='drug_fc1', num_hidden=num_hidden[-1])
    drug=mx.sym.Activation(data=drug, name='drug_relu1',act_type='tanh')

    #Cell part
    for h in num_hidden:
        cell=mx.sym.FullyConnected(data=cell, name='cell_fc_%d'%(h), num_hidden=h)
        cell=mx.sym.Activation(data=cell, name='cell_relu_%d'%(h),act_type='tanh')
        cell=mx.sym.Dropout(cell, p=0.2)
    pred=cell*drug
    #pred=mx.sym.sum_axis(data=pred, axis=1)
    #pred=mx.sym.Flatten(data=pred)
    pred=mx.sym.FullyConnected(data=pred, name = 'cls', num_hidden=1)
    pred=mx.sym.LinearRegressionOutput(data=pred, label = score)
    return pred

def getCID(cname):
    if cname not in cname2id:
        cname2id[cname] = len(cname2id)
    return cname2id[cname]

def getDID(dname):
    if dname not in dname2id:
        dname2id[dname]=len(dname2id)
    return dname2id[dname]

cell_file=sys.argv[1]
feedback_file = sys.argv[2]

feedback_data = []
cid2feature = collections.defaultdict(list)
cname2id = collections.defaultdict(int)
dname2id = collections.defaultdict(int)
num_features = 0
with open(cell_file, 'r') as f:
    for line in f:
        tks=line.strip().split('\t')
        cid=getCID(tks[0])
        features=map(float, tks[1:])
        if not num_features:
            num_features=len(features)
        cid2feature[cid]=features

with open(feedback_file, 'r') as f:
    for line in f:
        tks=line.strip().split('\t')
        if tks[0] not in cname2id:
            continue
        cid = getCID(tks[0])
        did = getDID(tks[1])
        v = float(tks[2])
        feedback_data.append([cid,did,v])
feedback_data=np.vstack(feedback_data)
print '#Cell, ', len(cname2id)
print '#Drug, ', len(dname2id)
print '#Features, ', num_features
print '#Scores, ', feedback_data.shape[0]
num_hidden=[512,256,128]
batch_size=500
num_epoch=200
learning_rate=0.0001
num_cell = len(cname2id)
num_drug = len(dname2id)
net=build_network(num_features, num_drug, num_hidden)
train(feedback_data, cid2feature, net, feedback_file, num_hidden, num_features, batch_size, num_epoch, learning_rate)

















