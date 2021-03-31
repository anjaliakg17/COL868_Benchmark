#!/usr/bin/env python
# coding: utf-8

# In[13]:


import json
from networkx.readwrite import json_graph
import os
import numpy as np
import networkx as nx
from pathlib import Path
import stellargraph as sg
from stellargraph.data import EdgeSplitter

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier 
#from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, StackingClassifier, VotingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score
from sklearn.metrics import precision_score, recall_score,f1_score
from sklearn.metrics.pairwise import cosine_similarity


# In[20]:


def edge_similarity(node1, node2):
    return np.abs((node1-node2))/2


# In[23]:


data_adj = np.loadtxt('PROTEINS_full_A.txt', delimiter=',').astype(int)
data_tuple = list(map(tuple, data_adj))

G = nx.Graph()
# add edges
G.add_edges_from(data_tuple)
for edge in G.edges():
    G[edge[0]][edge[1]]['weight'] = 1

G = G.to_undirected()


# In[51]:


node_labels = np.loadtxt('PROTEINS_full_node_labels.txt') 


# In[52]:


edges = [n for n in G.edges()]
protein_edge_file = "protein_edge_list.txt"
with open(protein_edge_file, 'w') as fp:
    fp.write('\n'.join('{} {}'.format(x[0],x[1]) for x in edges))


# In[55]:


N = 1000000
split_size = 5


# In[57]:



rep_sizes = [64, 128]
walk_lens = [80]
win_sizes = [10]
n_walks = [10]
for rep_size in rep_sizes:
    for walk_len in walk_lens:
        for win_size in win_sizes:
            for n_walk in n_walks:
                output_file = "protien_edge_embad_full_"+str(rep_size)+"_"+str(walk_len)+"_"+str(win_size)+"_"+str(n_walk)+".txt"
                command = "deepwalk --format edgelist --number-walks " + str(n_walk) + " --representation-size " + str(rep_size) + " --walk-length " + str(walk_len) + " --window-size " + str(win_size) + " --undirected true --input " + protein_edge_file + " --output " + output_file + " --workers 8"
                if not Path(output_file).is_file():
                    os.system(command)
                print(output_file)
                node_embeddings = np.loadtxt(output_file,skiprows=1)
                I1 = np.random.randint(0, node_embeddings.shape[0], size=(N))
                I2 = np.random.randint(0, node_embeddings.shape[0], size=(N))

                # Method 1
                X = edge_similarity(node_embeddings[I1], node_embeddings[I2])

                # Method 3
                # X_train = np.concatenate((node_embeddings[I1], node_embeddings[I2]), axis=1)


                y = node_labels[I1] == node_labels[I2]
                s = np.arange(X.shape[0])
                np.random.shuffle(s)
                X2 = X[s]
                y2 = y[s]
                roc = []
                prec = []
                rec = []
                f1 = []
                kf = KFold(n_splits=split_size)
                for train_index, test_index in kf.split(X2):
                    X_train2, X_test2 = X2[train_index], X2[test_index]
                    y_train2, y_test2 = y2[train_index], y2[test_index]
                    clf = RandomForestClassifier(random_state=1,verbose=1,n_estimators=5)
                    clf.fit(X_train2,y_train2)

                    pred = clf.predict(X_test2)


                    roc.append(roc_auc_score(y_test2,pred, average='micro'))
                    prec.append(precision_score(y_test2,pred, average='micro'))
                    rec.append(recall_score(y_test2,pred, average='micro'))
                    f1.append(f1_score(y_test2,pred, average='micro'))
                
                result = str(rep_size)+","+str(walk_len)+","+str(win_size)+","+str(n_walk)+","+str(np.mean(roc))+","+str(np.mean(prec))+","+str(np.mean(rec))+","+str(np.mean(f1))+"\n"
                f= open("result_pairwisenode_protein_deepwalk.txt","a+")
                f.write(result)
                f.close()


# In[58]:


rep_sizes = [128]
walk_lens = [10,200]
win_sizes = [10]
n_walks = [10]
for rep_size in rep_sizes:
    for walk_len in walk_lens:
        for win_size in win_sizes:
            for n_walk in n_walks:
                output_file = "protien_edge_embad_full_"+str(rep_size)+"_"+str(walk_len)+"_"+str(win_size)+"_"+str(n_walk)+".txt"
                command = "deepwalk --format edgelist --number-walks " + str(n_walk) + " --representation-size " + str(rep_size) + " --walk-length " + str(walk_len) + " --window-size " + str(win_size) + " --undirected true --input " + protein_edge_file + " --output " + output_file + " --workers 8"
                if not Path(output_file).is_file():
                    os.system(command)
                print(output_file)
                node_embeddings = np.loadtxt(output_file,skiprows=1)
                I1 = np.random.randint(0, node_embeddings.shape[0], size=(N))
                I2 = np.random.randint(0, node_embeddings.shape[0], size=(N))

                # Method 1
                X = edge_similarity(node_embeddings[I1], node_embeddings[I2])

                # Method 3
                # X_train = np.concatenate((node_embeddings[I1], node_embeddings[I2]), axis=1)


                y = node_labels[I1] == node_labels[I2]
                s = np.arange(X.shape[0])
                np.random.shuffle(s)
                X2 = X[s]
                y2 = y[s]
                roc = []
                prec = []
                rec = []
                f1 = []
                kf = KFold(n_splits=split_size)
                for train_index, test_index in kf.split(X2):
                    X_train2, X_test2 = X2[train_index], X2[test_index]
                    y_train2, y_test2 = y2[train_index], y2[test_index]
                    #clf = MLPClassifier(verbose=1)
                    #clf.fit(X_train2,y_train2)
                    clf = RandomForestClassifier(random_state=1,verbose=1,n_estimators=15)
                    clf.fit(X_train2,y_train2)

                    pred = clf.predict(X_test2)


                    roc.append(roc_auc_score(y_test2,pred, average='micro'))
                    prec.append(precision_score(y_test2,pred, average='micro'))
                    rec.append(recall_score(y_test2,pred, average='micro'))
                    f1.append(f1_score(y_test2,pred, average='micro'))
                
                result = str(rep_size)+","+str(walk_len)+","+str(win_size)+","+str(n_walk)+","+str(np.mean(roc))+","+str(np.mean(prec))+","+str(np.mean(rec))+","+str(np.mean(f1))+"\n"
                f= open("result_pairwisenode_protein_deepwalk.txt","a+")
                f.write(result)
                f.close()


# In[59]:



rep_sizes = [128]
walk_lens = [80]
win_sizes = [5,20]
n_walks = [10]
for rep_size in rep_sizes:
    for walk_len in walk_lens:
        for win_size in win_sizes:
            for n_walk in n_walks:
                output_file = "protien_edge_embad_full_"+str(rep_size)+"_"+str(walk_len)+"_"+str(win_size)+"_"+str(n_walk)+".txt"
                command = "deepwalk --format edgelist --number-walks " + str(n_walk) + " --representation-size " + str(rep_size) + " --walk-length " + str(walk_len) + " --window-size " + str(win_size) + " --undirected true --input " + protein_edge_file + " --output " + output_file + " --workers 8"
                if not Path(output_file).is_file():
                    os.system(command)
                print(output_file)
                node_embeddings = np.loadtxt(output_file,skiprows=1)
                I1 = np.random.randint(0, node_embeddings.shape[0], size=(N))
                I2 = np.random.randint(0, node_embeddings.shape[0], size=(N))

                # Method 1
                X = edge_similarity(node_embeddings[I1], node_embeddings[I2])

                # Method 3
                # X_train = np.concatenate((node_embeddings[I1], node_embeddings[I2]), axis=1)


                y = node_labels[I1] == node_labels[I2]
                s = np.arange(X.shape[0])
                np.random.shuffle(s)
                X2 = X[s]
                y2 = y[s]
                roc = []
                prec = []
                rec = []
                f1 = []
                kf = KFold(n_splits=split_size)
                for train_index, test_index in kf.split(X2):
                    X_train2, X_test2 = X2[train_index], X2[test_index]
                    y_train2, y_test2 = y2[train_index], y2[test_index]
                    #clf = MLPClassifier(verbose=1)
                    #clf.fit(X_train2,y_train2)
                    clf = RandomForestClassifier(random_state=1,verbose=1,n_estimators=15)
                    clf.fit(X_train2,y_train2)

                    pred = clf.predict(X_test2)


                    roc.append(roc_auc_score(y_test2,pred, average='micro'))
                    prec.append(precision_score(y_test2,pred, average='micro'))
                    rec.append(recall_score(y_test2,pred, average='micro'))
                    f1.append(f1_score(y_test2,pred, average='micro'))
                
                result = str(rep_size)+","+str(walk_len)+","+str(win_size)+","+str(n_walk)+","+str(np.mean(roc))+","+str(np.mean(prec))+","+str(np.mean(rec))+","+str(np.mean(f1))+"\n"
                f= open("result_pairwisenode_protein_deepwalk.txt","a+")
                f.write(result)
                f.close()


# In[60]:


rep_sizes = [128]
walk_lens = [80]
win_sizes = [10]
n_walks = [5,50]
for rep_size in rep_sizes:
    for walk_len in walk_lens:
        for win_size in win_sizes:
            for n_walk in n_walks:
                output_file = "protien_edge_embad_full_"+str(rep_size)+"_"+str(walk_len)+"_"+str(win_size)+"_"+str(n_walk)+".txt"
                command = "deepwalk --format edgelist --number-walks " + str(n_walk) + " --representation-size " + str(rep_size) + " --walk-length " + str(walk_len) + " --window-size " + str(win_size) + " --undirected true --input " + protein_edge_file + " --output " + output_file + " --workers 8"
                if not Path(output_file).is_file():
                    os.system(command)
                print(output_file)
                node_embeddings = np.loadtxt(output_file,skiprows=1)
                I1 = np.random.randint(0, node_embeddings.shape[0], size=(N))
                I2 = np.random.randint(0, node_embeddings.shape[0], size=(N))

                # Method 1
                X = edge_similarity(node_embeddings[I1], node_embeddings[I2])

                # Method 3
                # X_train = np.concatenate((node_embeddings[I1], node_embeddings[I2]), axis=1)


                y = node_labels[I1] == node_labels[I2]
                s = np.arange(X.shape[0])
                np.random.shuffle(s)
                X2 = X[s]
                y2 = y[s]
                roc = []
                prec = []
                rec = []
                f1 = []
                kf = KFold(n_splits=split_size)
                for train_index, test_index in kf.split(X2):
                    X_train2, X_test2 = X2[train_index], X2[test_index]
                    y_train2, y_test2 = y2[train_index], y2[test_index]
                    #clf = MLPClassifier(verbose=1)
                    #clf.fit(X_train2,y_train2)
                    clf = RandomForestClassifier(random_state=1,verbose=1,n_estimators=15)
                    clf.fit(X_train2,y_train2)

                    pred = clf.predict(X_test2)


                    roc.append(roc_auc_score(y_test2,pred, average='micro'))
                    prec.append(precision_score(y_test2,pred, average='micro'))
                    rec.append(recall_score(y_test2,pred, average='micro'))
                    f1.append(f1_score(y_test2,pred, average='micro'))
                
                result = str(rep_size)+","+str(walk_len)+","+str(win_size)+","+str(n_walk)+","+str(np.mean(roc))+","+str(np.mean(prec))+","+str(np.mean(rec))+","+str(np.mean(f1))+"\n"
                f= open("result_pairwisenode_protein_deepwalk.txt","a+")
                f.write(result)
                f.close()


# In[ ]:




