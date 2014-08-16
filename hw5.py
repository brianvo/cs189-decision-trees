import scipy.io
import numpy as np
from math import log

class Node(object):
  attr = -1
  side = -1
  val = ""
  left = None
  right = None
  def __init__(self, attr, side, val, left, right):
    self.attr = attr
    self.side = side
    self.val = val
    self.left = left
    self.right = right

class SpamSet(object):
  x = None
  y = None
  def __init__(self, x, y):
    self.x = x
    self.y = y

spam = scipy.io.loadmat('spam.mat')
Xtrain = spam['Xtrain']
ytrain = spam['ytrain']
Xtest = spam['Xtest']

x_size = len(Xtrain[1])

#Computes the mean of means of an attribute
def mean_of_means(S,a):
  num_zeros = 0
  zero_mean = 0
  num_ones = 0
  one_mean = 0
  for i in range(len(S.x)):
    if S.y[i] == 0:
      zero_mean = zero_mean + S.x[i][a]
      num_zeros = num_zeros + 1
    else:
      one_mean = one_mean + S.x[i][a]
      num_ones = num_ones + 1
  zero_mean = (zero_mean * 1.0)/(num_zeros * 1.0)
  one_mean = (one_mean * 1.0)/(num_ones * 1.0)
  if zero_mean <= one_mean:
    return ((zero_mean + one_mean)/2, 0)
  else:
    return ((zero_mean + one_mean)/2, 1)  

#Splits a set on an attribute
def split_set(S, a):
  mean = mean_of_means(S,a)
  u = mean[0]
  side = mean[1]
  
  S_0_indexes = list()
  S_1_indexes = list()
  for i in range(len(S.x)):
    if side == 0:
      if S.x[i][a] <= u:
        S_0_indexes.append(i)
      else:
        S_1_indexes.append(i)
    else:
      if S.x[i][a] >= u:
        S_0_indexes.append(i)
      else:
        S_1_indexes.append(i)

  S_0_x = np.zeros(shape=(len(S_0_indexes),x_size))
  S_0_y = np.zeros(shape=(len(S_0_indexes),1))
  for i in range(len(S_0_indexes)):
    S_0_x[i] = S.x[S_0_indexes[i]]
    S_0_y[i] = S.y[S_0_indexes[i]]
    
  S_1_x = np.zeros(shape=(len(S_1_indexes),x_size))
  S_1_y = np.zeros(shape=(len(S_1_indexes),1))
  for i in range(len(S_1_indexes)):
    S_1_x[i] = S.x[S_1_indexes[i]]
    S_1_y[i] = S.y[S_1_indexes[i]]
    
  S_0 = SpamSet(S_0_x, S_0_y)
  S_1 = SpamSet(S_1_x, S_1_y)
  return (S_0, S_1)

#Gets the entropy of a set
def get_entropy(S):
  if len(S.y) == 0:
    return 0
  num_zeros = 0
  for y in S.y:
    if y == 0:
      num_zeros = num_zeros + 1
  P = (num_zeros*1.0)/(len(S.y)*1.0)
  if P == 0.0 or P == 1.0:
    return 0
  return -1*P*log(P) + -1*(1-P)*log(1-P)

#Finds the best attribute to create the next node on
def best_attribute(S,depth):
  max_information = 0
  best_a = -1
  H = get_entropy(S)
  for a in range(x_size):
    sets = split_set(S,a)
    H_0 = get_entropy(sets[0])
    H_1 = get_entropy(sets[1])
    information = H - (len(sets[0].x)*1.0)/(len(S.x)*1.0)*H_0 - (len(sets[1].x)*1.0)/(len(S.x)*1.0)*H_1
    if information >= max_information:
      max_information = information
      best_a = a
  return best_a

#Tests if all x data points in S are the same, avoids getting stuck with same x data points but different y values
def all_same_x(S):
  data = S.x[0]
  num_zeros = 0
  for i in range(len(S.x)):
    if (S.x[i] == data).all():
      if S.y[i] == 0:
        num_zeros = num_zeros + 1
    else:
      return -1
  if num_zeros >= len(S.x) - num_zeros:
    return 0
  else:
    return 1

#Constructs the decision tree
def grow_tree(S, depth):
  all_same_sum = 0
  for i in range(len(S.y)):
    all_same_sum += S.y[i]
  if all_same_sum == 0:
    return Node(-1, -1, 0, None, None)
  elif all_same_sum == len(S.y):
    return Node(-1, -1, 1, None, None)    
  else:
    same_x = all_same_x(S)
    if same_x == 0:
      return Node(-1, -1, 0, None, None)
    elif same_x == 1:
      return Node(-1, -1, 1, None, None)
    a = best_attribute(S,depth)
    if a == -1:
      return Node(-1, -1, 0, None, None)
    #print str(depth) + " , " + str(a)
    sets = split_set(S,a)
    mean = mean_of_means(S,a)
    return Node(a, mean[1], mean[0], grow_tree(sets[0], depth + 1), grow_tree(sets[1], depth + 1))        

#Testing goes here

indexes = np.random.permutation(Xtrain.shape[0])
k = 10
for i in range(k):
  S_x = np.zeros(shape=(Xtrain.shape[0]*(k-1)/k,x_size))
  S_y = np.zeros(shape=(Xtrain.shape[0]*(k-1)/k,1))
  test_x = np.zeros(shape=(Xtrain.shape[0]/k,x_size))
  test_y = np.zeros(shape=(Xtrain.shape[0]/k,1))
  S_index = 0
  test_index = 0
  for j in range(Xtrain.shape[0]):
    if not (j >= i*(Xtrain.shape[0]/k) and j < (i+1)*(Xtrain.shape[0]/k)):
      S_x[S_index] = Xtrain[indexes[j]]
      S_y[S_index] = ytrain[indexes[j]]
      S_index = S_index + 1
    else:
      test_x[test_index] = Xtrain[indexes[j]]
      test_y[test_index] = ytrain[indexes[j]]
      test_index = test_index + 1
  S = SpamSet(S_x, S_y)
  
  root = grow_tree(S,0)
  
  yvals = list()
  for x in test_x:
    current_node = root
    while current_node.attr != -1:
      if current_node.side == 0:
        if x[current_node.attr] <= current_node.val:
          current_node = current_node.left
        else:
          current_node = current_node.right
      elif current_node.side == 1:
        if x[current_node.attr] >= current_node.val:
          current_node = current_node.left
        else:
          current_node = current_node.right
    yvals.append(current_node.val)

  num_errors = 0
  for i in range(len(test_y)):
    if test_y[i] != yvals[i]:
      num_errors = num_errors + 1
  #print (num_errors*1.0)/(len(test_y)*1.0)
