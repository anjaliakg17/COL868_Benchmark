#!/usr/bin/env python
# coding: utf-8

# In[3]:


import json
from networkx.readwrite import json_graph
import os
import numpy as np
import networkx as nx

import stellargraph as sg
from stellargraph.data import EdgeSplitter


# In[4]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier 
#from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, StackingClassifier, VotingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score
from sklearn.metrics import precision_score, recall_score,f1_score


# In[5]:


G=nx.read_edgelist('loc-brightkite_edges.txt')


# In[6]:


print(nx.info(G))


# In[7]:


G = G.to_undirected()


# In[82]:


edge_splitter_test = EdgeSplitter(G)
test_size = 0.7
G2, edge_ids_test2, edge_labels_test2 = edge_splitter_test.train_test_split(
    p=test_size, method="global", keep_connected=True
)


# In[84]:


n2 = int(edge_ids_test2.shape[0])


# In[87]:


edge_splitter_test = EdgeSplitter(G)
test_size = 0.5
G_train, edge_ids_test_temp, edge_labels_test_temp = edge_splitter_test.train_test_split(
    p=test_size, method="global", keep_connected=True
)


# In[89]:


neg_sample_size = int(edge_ids_test_temp.shape[0]/2)


# In[110]:


train_pos_edges_cnt = len(G_train.edges())
edge_ids_train = np.zeros((2*train_pos_edges_cnt,2))
edge_labels_train = np.zeros((2*train_pos_edges_cnt,))

edge_ids_train[0:train_pos_edges_cnt] = np.array(G_train.edges())
edge_ids_train[train_pos_edges_cnt+1:] = edge_ids_test2[-neg_sample_size+1:]

edge_labels_train[0:train_pos_edges_cnt] = edge_labels_test2[0:train_pos_edges_cnt]


# In[78]:


test_size_final = (int(n2/2)-train_pos_edges_cnt)*2
edge_ids_test = np.zeros((test_size_final,2))
edge_labels_test = np.zeros((test_size_final,))
edge_ids_test[0:int(test_size_final/2)] = edge_ids_test_temp[0:int(test_size_final/2)]
edge_ids_test[int(test_size_final/2)+1:] = edge_ids_test2[int(n2/2)+1:n2-neg_sample_size]
edge_labels_test[0:int(test_size_final/2)] = edge_labels_test_temp[0:int(test_size_final/2)]


# In[92]:


edges = [n for n in G_train.edges()]
brightkite_edge_file_train = "brightkite_edge_list_train.txt"
with open(brightkite_edge_file_train, 'w') as fp:
    fp.write('\n'.join('{} {}'.format(x[0],x[1]) for x in edges))


# In[111]:


brightkite_edge_file_test = "brightkite_edge_list_test_with_pos_neg.txt"
with open(brightkite_edge_file_test, 'w') as fp:
    fp.write('\n'.join('{} {}'.format(x[0],x[1]) for x in edge_ids_test))


# In[94]:


rep_size = 64
walk_len = 40
win_size = 5
n_walk = 10 
output_file = "brightkite_embad.txt"


# In[95]:


command = "deepwalk --format edgelist --number-walks " + str(n_walk) + " --representation-size " + str(rep_size) + " --walk-length " + str(walk_len) + " --window-size " + str(win_size) + " --undirected true --input " + brightkite_edge_file_train + " --output " + output_file + " --workers 8"


# In[96]:


os.system(command)


# In[97]:


data_emb = np.loadtxt(output_file,skiprows=1)
emb_dim = len(data_emb[0])-1


# In[98]:


embedding = np.zeros((len(G_train.nodes()),emb_dim))


# In[99]:


for idx in range(data_emb.shape[0]):
    embedding[int(data_emb[idx][0])] = data_emb[idx][1:]


# In[115]:


X = np.zeros((edge_ids_train.shape[0],emb_dim))
y = edge_labels_train
X_test = np.zeros((edge_ids_test.shape[0],emb_dim))
y_test = edge_labels_test


# In[116]:


def agg_fun1(x,y):
    return (x+y)/2
def agg_fun2(x,y):
    return (x*y)/2
def agg_fun3(x,y):
    return np.abs(x-y)
def agg_fun4(x,y):
    return np.square(x-y)


# In[138]:


choice = 4


# In[139]:


idx = 0
for e in edge_ids_train:
    if choice == 1:
        X[idx] = agg_fun1(embedding[int(e[0])], embedding[int(e[1])])
    if choice == 2:
        X[idx] = agg_fun2(embedding[int(e[0])], embedding[int(e[1])])
    if choice == 3:
        X[idx] = agg_fun3(embedding[int(e[0])], embedding[int(e[1])])
    if choice == 4:
        X[idx] = agg_fun4(embedding[int(e[0])], embedding[int(e[1])])
    idx += 1
    


# In[140]:


s = np.arange(X.shape[0])
np.random.shuffle(s)
X_train = X[s]
y_train = y[s]


# In[141]:


idx = 0
for e in edge_ids_test:
    if choice == 1:
        X_test[idx] = agg_fun1(embedding[int(e[0])], embedding[int(e[1])])
    if choice == 2:
        X_test[idx] = agg_fun2(embedding[int(e[0])], embedding[int(e[1])])
    if choice == 3:
        X_test[idx] = agg_fun3(embedding[int(e[0])], embedding[int(e[1])])
    if choice == 4:
        X_test[idx] = agg_fun4(embedding[int(e[0])], embedding[int(e[1])])
        
    idx += 1
    


# In[142]:


clf = LogisticRegression(penalty='l2', solver='saga',class_weight='balanced')
clf.fit(X_train,y_train)
pred = clf.predict(X_test)


# In[143]:


roc = roc_auc_score(y_test,pred)
prec = precision_score(y_test,pred)
rec = recall_score(y_test,pred)
f1 = f1_score(y_test,pred)


# In[144]:


result = str(roc)+","+str(prec)+","+str(rec)+","+str(f1)+", agg_fun"+str(choice)+"\n"
f= open("result_linkpred_brightkite_deepwalk.txt","a+")
f.write(result)
f.close()
print(result)


# In[57]:


s = np.arange(X.shape[0])
np.random.shuffle(s)
X2 = X[s]
y2 = y[s]
roc = []
prec = []
rec = []
f1 = []
kf = KFold(n_splits=5)
for train_index, test_index in kf.split(X2):
    X_train2, X_test2 = X2[train_index], X2[test_index]
    y_train2, y_test2 = y2[train_index], y2[test_index]
    clf = LogisticRegression(penalty='l2', solver='saga',class_weight='balanced')
    clf.fit(X_train2,y_train2)
    pred = clf.predict(X_test)
        
    roc.append(roc_auc_score(y_test,pred))
    prec.append(precision_score(y_test,pred))
    rec.append(recall_score(y_test,pred))
    f1.append(f1_score(y_test,pred))


# In[58]:


print(np.mean(roc),np.mean(prec),np.mean(rec),np.mean(f1))


# In[59]:


X_comb = np.row_stack((X,X_test))
y_comb = np.array(list(y) + list(y_test))
s = np.arange(X_comb.shape[0])
np.random.shuffle(s)
X2 = X_comb[s]
y2 = y_comb[s]
roc = []
prec = []
rec = []
f1 = []
kf = KFold(n_splits=5)
for train_index, test_index in kf.split(X2):
    X_train2, X_test2 = X2[train_index], X2[test_index]
    y_train2, y_test2 = y2[train_index], y2[test_index]
    clf = LogisticRegression(penalty='l2', solver='saga',class_weight='balanced')
    clf.fit(X_train2,y_train2)
    pred = clf.predict(X_test2)
        
    roc.append(roc_auc_score(y_test2,pred))
    prec.append(precision_score(y_test2,pred))
    rec.append(recall_score(y_test2,pred))
    f1.append(f1_score(y_test2,pred))


# In[60]:


print(np.mean(roc),np.mean(prec),np.mean(rec),np.mean(f1))


# In[ ]:




