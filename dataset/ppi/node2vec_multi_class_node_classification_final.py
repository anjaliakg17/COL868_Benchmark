#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from node2vec import Node2Vec


# In[2]:


from sklearn.naive_bayes import GaussianNB 
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier


# In[3]:


G = json_graph.node_link_graph(json.load(open("ppi-G.json")))
edges = [n for n in G.edges()]
ppi_edge_file = "ppi_edge_list.txt"
with open(ppi_edge_file, 'w') as fp:
    fp.write('\n'.join('{} {}'.format(x[0],x[1]) for x in edges))


# In[4]:


class_map = json.load(open("ppi-class_map.json"))
n = len(class_map.keys())
m = len(class_map['0'])
target = np.zeros((n,m))
for i in range(n):
    target[i] = np.array(class_map[str(i)])


# In[5]:


edge_labels_internal = json.load(open("ppi-class_map.json"))
edge_labels_internal = {int(i): l for i, l in edge_labels_internal.items()}
train_ids = [n for n in G.nodes()]
train_labels = np.array([edge_labels_internal[i] for i in train_ids])
if train_labels.ndim == 1:
    train_labels = np.expand_dims(train_labels, 1)


# In[6]:


split_size = 2


# In[7]:


directory = "tmp"
if not os.path.exists(directory):
    os.makedirs(directory)


# In[9]:


EMBEDDING_DIMS = [64, 128]
WALK_LENGTHS = [40]
NUM_WALKS = [10]
WORKERS = [4]
Ps = [1]
Qs = [1]
for EMBEDDING_DIM in EMBEDDING_DIMS:
    for WALK_LENGTH in WALK_LENGTHS:
        for NUM_WALK in NUM_WALKS:
            for WORKER in WORKERS:
                for P in Ps:
                    for Q in Qs:
                        filename = "ppi_node2vec_full_embeddings_"+str(EMBEDDING_DIM)+"_"+str(WALK_LENGTH)+"_"+str(NUM_WALK)+"_"+str(P)+"_"+str(Q)+".emb"
                        print(filename)
                        if not Path(filename).is_file():
                            node2vec = Node2Vec(G, dimensions=EMBEDDING_DIM, walk_length=WALK_LENGTH, num_walks=NUM_WALK, workers=WORKER, p = P, q = Q, temp_folder="tmp/")
                            model = node2vec.fit(window=10, min_count=1, batch_words=4)
                            model.wv.save_word2vec_format(filename)
                        
                        data_emb = np.loadtxt(filename,skiprows=1)
                        emb_dim = len(data_emb[0])-1
                        num_nodes = len(list(G.nodes()))
                        embedding = np.zeros((len(G.nodes()),emb_dim))
                        for idx in range(data_emb.shape[0]):
                            embedding[int(data_emb[idx][0])] = data_emb[idx][1:]
                        X = np.zeros((num_nodes,emb_dim))
                        idx = 0
                        for node in G.nodes():
                            X[idx] = embedding[node]
                            idx += 1

                        y = target
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
                            #clf = MLPClassifier(verbose=1)
                            #clf.fit(X_train2,y_train2)
                            forest = RandomForestClassifier(random_state=1,verbose=1,n_estimators=10)
                            clf = MultiOutputClassifier(forest, n_jobs=-1)
                            clf.fit(X_train2,y_train2)

                            pred = clf.predict(X_test2)


                            roc.append(roc_auc_score(y_test2,pred, average='micro'))
                            prec.append(precision_score(y_test2,pred, average='micro'))
                            rec.append(recall_score(y_test2,pred, average='micro'))
                            f1.append(f1_score(y_test2,pred, average='micro'))

                        result = str(EMBEDDING_DIM)+","+str(WALK_LENGTH)+","+str(NUM_WALK)+","+str(P)+","+str(Q)+","+str(np.mean(roc))+","+str(np.mean(prec))+","+str(np.mean(rec))+","+str(np.mean(f1))+"\n"
                        f= open("result_multiclass_ppi_node2vec.txt","a+")
                        f.write(result)
                        f.close()
                        
                        


# In[8]:


EMBEDDING_DIMS = [64]
WALK_LENGTHS = [10,100]
NUM_WALKS = [10]
WORKERS = [4]
Ps = [1]
Qs = [1]
for EMBEDDING_DIM in EMBEDDING_DIMS:
    for WALK_LENGTH in WALK_LENGTHS:
        for NUM_WALK in NUM_WALKS:
            for WORKER in WORKERS:
                for P in Ps:
                    for Q in Qs:
                        filename = "ppi_node2vec_full_embeddings_"+str(EMBEDDING_DIM)+"_"+str(WALK_LENGTH)+"_"+str(NUM_WALK)+"_"+str(P)+"_"+str(Q)+".emb"
                        print(filename)
                        if not Path(filename).is_file():
                            node2vec = Node2Vec(G, dimensions=EMBEDDING_DIM, walk_length=WALK_LENGTH, num_walks=NUM_WALK, workers=WORKER, p = P, q = Q, temp_folder="tmp/")
                            model = node2vec.fit(window=10, min_count=1, batch_words=4)
                            model.wv.save_word2vec_format(filename)
                        
                        data_emb = np.loadtxt(filename,skiprows=1)
                        emb_dim = len(data_emb[0])-1
                        num_nodes = len(list(G.nodes()))
                        embedding = np.zeros((len(G.nodes()),emb_dim))
                        for idx in range(data_emb.shape[0]):
                            embedding[int(data_emb[idx][0])] = data_emb[idx][1:]
                        X = np.zeros((num_nodes,emb_dim))
                        idx = 0
                        for node in G.nodes():
                            X[idx] = embedding[node]
                            idx += 1

                        y = target
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
                            #clf = MLPClassifier(verbose=1)
                            #clf.fit(X_train2,y_train2)
                            forest = RandomForestClassifier(random_state=1,verbose=1,n_estimators=10)
                            clf = MultiOutputClassifier(forest, n_jobs=-1)
                            clf.fit(X_train2,y_train2)

                            pred = clf.predict(X_test2)


                            roc.append(roc_auc_score(y_test2,pred, average='micro'))
                            prec.append(precision_score(y_test2,pred, average='micro'))
                            rec.append(recall_score(y_test2,pred, average='micro'))
                            f1.append(f1_score(y_test2,pred, average='micro'))

                        result = str(EMBEDDING_DIM)+","+str(WALK_LENGTH)+","+str(NUM_WALK)+","+str(P)+","+str(Q)+","+str(np.mean(roc))+","+str(np.mean(prec))+","+str(np.mean(rec))+","+str(np.mean(f1))+"\n"
                        f= open("result_multiclass_ppi_node2vec.txt","a+")
                        f.write(result)
                        f.close()
                        
                        


# In[10]:


EMBEDDING_DIMS = [64]
WALK_LENGTHS = [40]
NUM_WALKS = [20,50]
WORKERS = [4]
Ps = [1]
Qs = [1]
for EMBEDDING_DIM in EMBEDDING_DIMS:
    for WALK_LENGTH in WALK_LENGTHS:
        for NUM_WALK in NUM_WALKS:
            for WORKER in WORKERS:
                for P in Ps:
                    for Q in Qs:
                        filename = "ppi_node2vec_full_embeddings_"+str(EMBEDDING_DIM)+"_"+str(WALK_LENGTH)+"_"+str(NUM_WALK)+"_"+str(P)+"_"+str(Q)+".emb"
                        print(filename)
                        if not Path(filename).is_file():
                            node2vec = Node2Vec(G, dimensions=EMBEDDING_DIM, walk_length=WALK_LENGTH, num_walks=NUM_WALK, workers=WORKER, p = P, q = Q, temp_folder="tmp/")
                            model = node2vec.fit(window=10, min_count=1, batch_words=4)
                            model.wv.save_word2vec_format(filename)
                        
                        data_emb = np.loadtxt(filename,skiprows=1)
                        emb_dim = len(data_emb[0])-1
                        num_nodes = len(list(G.nodes()))
                        embedding = np.zeros((len(G.nodes()),emb_dim))
                        for idx in range(data_emb.shape[0]):
                            embedding[int(data_emb[idx][0])] = data_emb[idx][1:]
                        X = np.zeros((num_nodes,emb_dim))
                        idx = 0
                        for node in G.nodes():
                            X[idx] = embedding[node]
                            idx += 1

                        y = target
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
                            #clf = MLPClassifier(verbose=1)
                            #clf.fit(X_train2,y_train2)
                            forest = RandomForestClassifier(random_state=1,verbose=1,n_estimators=10)
                            clf = MultiOutputClassifier(forest, n_jobs=-1)
                            clf.fit(X_train2,y_train2)

                            pred = clf.predict(X_test2)


                            roc.append(roc_auc_score(y_test2,pred, average='micro'))
                            prec.append(precision_score(y_test2,pred, average='micro'))
                            rec.append(recall_score(y_test2,pred, average='micro'))
                            f1.append(f1_score(y_test2,pred, average='micro'))

                        result = str(EMBEDDING_DIM)+","+str(WALK_LENGTH)+","+str(NUM_WALK)+","+str(P)+","+str(Q)+","+str(np.mean(roc))+","+str(np.mean(prec))+","+str(np.mean(rec))+","+str(np.mean(f1))+"\n"
                        f= open("result_multiclass_ppi_node2vec.txt","a+")
                        f.write(result)
                        f.close()
                        
                        


# In[8]:


EMBEDDING_DIMS = [64]
WALK_LENGTHS = [40]
NUM_WALKS = [10]
WORKERS = [4]
Ps = [0.2, 2]
Qs = [1]
for EMBEDDING_DIM in EMBEDDING_DIMS:
    for WALK_LENGTH in WALK_LENGTHS:
        for NUM_WALK in NUM_WALKS:
            for WORKER in WORKERS:
                for P in Ps:
                    for Q in Qs:
                        filename = "ppi_node2vec_full_embeddings_"+str(EMBEDDING_DIM)+"_"+str(WALK_LENGTH)+"_"+str(NUM_WALK)+"_"+str(P)+"_"+str(Q)+".emb"
                        print(filename)
                        if not Path(filename).is_file():
                            node2vec = Node2Vec(G, dimensions=EMBEDDING_DIM, walk_length=WALK_LENGTH, num_walks=NUM_WALK, workers=WORKER, p = P, q = Q, temp_folder="tmp/")
                            model = node2vec.fit(window=10, min_count=1, batch_words=4)
                            model.wv.save_word2vec_format(filename)
                        
                        data_emb = np.loadtxt(filename,skiprows=1)
                        emb_dim = len(data_emb[0])-1
                        num_nodes = len(list(G.nodes()))
                        embedding = np.zeros((len(G.nodes()),emb_dim))
                        for idx in range(data_emb.shape[0]):
                            embedding[int(data_emb[idx][0])] = data_emb[idx][1:]
                        X = np.zeros((num_nodes,emb_dim))
                        idx = 0
                        for node in G.nodes():
                            X[idx] = embedding[node]
                            idx += 1

                        y = target
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
                            #clf = MLPClassifier(verbose=1)
                            #clf.fit(X_train2,y_train2)
                            forest = RandomForestClassifier(random_state=1,verbose=1,n_estimators=10)
                            clf = MultiOutputClassifier(forest, n_jobs=-1)
                            clf.fit(X_train2,y_train2)

                            pred = clf.predict(X_test2)


                            roc.append(roc_auc_score(y_test2,pred, average='micro'))
                            prec.append(precision_score(y_test2,pred, average='micro'))
                            rec.append(recall_score(y_test2,pred, average='micro'))
                            f1.append(f1_score(y_test2,pred, average='micro'))

                        result = str(EMBEDDING_DIM)+","+str(WALK_LENGTH)+","+str(NUM_WALK)+","+str(P)+","+str(Q)+","+str(np.mean(roc))+","+str(np.mean(prec))+","+str(np.mean(rec))+","+str(np.mean(f1))+"\n"
                        f= open("result_multiclass_ppi_node2vec.txt","a+")
                        f.write(result)
                        f.close()
                        
                        
                        


# In[9]:


EMBEDDING_DIMS = [64]
WALK_LENGTHS = [40]
NUM_WALKS = [10]
WORKERS = [4]
Ps = [1]
Qs = [0.2, 2]

for EMBEDDING_DIM in EMBEDDING_DIMS:
    for WALK_LENGTH in WALK_LENGTHS:
        for NUM_WALK in NUM_WALKS:
            for WORKER in WORKERS:
                for P in Ps:
                    for Q in Qs:
                        filename = "ppi_node2vec_full_embeddings_"+str(EMBEDDING_DIM)+"_"+str(WALK_LENGTH)+"_"+str(NUM_WALK)+"_"+str(P)+"_"+str(Q)+".emb"
                        print(filename)
                        if not Path(filename).is_file():
                            node2vec = Node2Vec(G, dimensions=EMBEDDING_DIM, walk_length=WALK_LENGTH, num_walks=NUM_WALK, workers=WORKER, p = P, q = Q, temp_folder="tmp/")
                            model = node2vec.fit(window=10, min_count=1, batch_words=4)
                            model.wv.save_word2vec_format(filename)
                        
                        data_emb = np.loadtxt(filename,skiprows=1)
                        emb_dim = len(data_emb[0])-1
                        num_nodes = len(list(G.nodes()))
                        embedding = np.zeros((len(G.nodes()),emb_dim))
                        for idx in range(data_emb.shape[0]):
                            embedding[int(data_emb[idx][0])] = data_emb[idx][1:]
                        X = np.zeros((num_nodes,emb_dim))
                        idx = 0
                        for node in G.nodes():
                            X[idx] = embedding[node]
                            idx += 1

                        y = target
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
                            #clf = MLPClassifier(verbose=1)
                            #clf.fit(X_train2,y_train2)
                            forest = RandomForestClassifier(random_state=1,verbose=1,n_estimators=10)
                            clf = MultiOutputClassifier(forest, n_jobs=-1)
                            clf.fit(X_train2,y_train2)

                            pred = clf.predict(X_test2)


                            roc.append(roc_auc_score(y_test2,pred, average='micro'))
                            prec.append(precision_score(y_test2,pred, average='micro'))
                            rec.append(recall_score(y_test2,pred, average='micro'))
                            f1.append(f1_score(y_test2,pred, average='micro'))

                        result = str(EMBEDDING_DIM)+","+str(WALK_LENGTH)+","+str(NUM_WALK)+","+str(P)+","+str(Q)+","+str(np.mean(roc))+","+str(np.mean(prec))+","+str(np.mean(rec))+","+str(np.mean(f1))+"\n"
                        f= open("result_multiclass_ppi_node2vec.txt","a+")
                        f.write(result)
                        f.close()
                        
                        
                        


# In[ ]:




