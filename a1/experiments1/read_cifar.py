import cPickle as pickle
import numpy as np
import os

def change(inputArray):
  result = []
  for i in inputArray:
    temp = [0] * 10
    temp[i] = 1
    result.append(temp)
  result = np.array(result,dtype=np.float32)
  return result


def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'r') as f:
    datadict = pickle.load(f)
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y

def load_CIFAR10(ROOT):
  """ load all of cifar """
  xs = []
  ys = []
  for b in range(1,5):
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)    
  Xtr = np.concatenate(xs)
  Xtr = np.array(Xtr,dtype=np.float32)
  Ytr = np.concatenate(ys)
  Ytr = change(Ytr)
  del X, Y

  xs = []
  ys = []
  for b in range(5,6):
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)    
  Xva = np.concatenate(xs)
  Xva = np.array(Xva,dtype=np.float32)
  Yva = np.concatenate(ys)
  Yva = change(Yva)
  del X, Y

  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
  Xte = np.array(Xte,dtype=np.float32)
  Yte = change(Yte)
  return (Xtr, Ytr), (Xva, Yva), (Xte, Yte)