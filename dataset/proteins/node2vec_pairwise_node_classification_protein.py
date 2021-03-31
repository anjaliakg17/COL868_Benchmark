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


def edge_similarity(node1, node2):
    return np.abs((node1-node2))/2


# In[3]:



data_adj = np.loadtxt('PROTEINS_full_A.txt', delimiter=',').astype(int)
data_tuple = list(map(tuple, data_adj))

G = nx.Graph()
# add edges
G.add_edges_from(data_tuple)
for edge in G.edges():
    G[edge[0]][edge[1]]['weight'] = 1

G = G.to_undirected()


# In[4]:


node_labels = np.loadtxt('PROTEINS_full_node_labels.txt') 


# In[5]:


edges = [n for n in G.edges()]
protein_edge_file = "protein_edge_list.txt"
with open(protein_edge_file, 'w') as fp:
    fp.write('\n'.join('{} {}'.format(x[0],x[1]) for x in edges))


# In[11]:


N = 1000000
split_size = 5
n_estimators_num = 10


# In[8]:


directory = "tmp"
if not os.path.exists(directory):
    os.makedirs(directory)


# In[10]:


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
                        filename = "protein_node2vec_full_embeddings_"+str(EMBEDDING_DIM)+"_"+str(WALK_LENGTH)+"_"+str(NUM_WALK)+"_"+str(P)+"_"+str(Q)+".emb"
                        print(filename)
                        if not Path(filename).is_file():
                            node2vec = Node2Vec(G, dimensions=EMBEDDING_DIM, walk_length=WALK_LENGTH, num_walks=NUM_WALK, workers=WORKER, p = P, q = Q, temp_folder="tmp/")
                            model = node2vec.fit(window=10, min_count=1, batch_words=4)
                            model.wv.save_word2vec_format(filename)
                        node_embeddings = np.loadtxt(filename,skiprows=1)
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
                        kf = KFold(n_splits=5)
                        for train_index, test_index in kf.split(X2):
                            X_train2, X_test2 = X2[train_index], X2[test_index]
                            y_train2, y_test2 = y2[train_index], y2[test_index]
                            #clf = MLPClassifier(verbose=1)
                            #clf.fit(X_train2,y_train2)
                            clf = RandomForestClassifier(random_state=1,verbose=1,n_estimators=n_estimators_num)
                            clf.fit(X_train2,y_train2)

                            pred = clf.predict(X_test2)


                            roc.append(roc_auc_score(y_test2,pred, average='micro'))
                            prec.append(precision_score(y_test2,pred, average='micro'))
                            rec.append(recall_score(y_test2,pred, average='micro'))
                            f1.append(f1_score(y_test2,pred, average='micro'))

                        result = str(EMBEDDING_DIM)+","+str(WALK_LENGTH)+","+str(NUM_WALK)+","+str(P)+","+str(Q)+","+str(np.mean(roc))+","+str(np.mean(prec))+","+str(np.mean(rec))+","+str(np.mean(f1))+"\n"
                        f= open("result_pairwisenode_protein_node2vec.txt","a+")
                        f.write(result)
                        f.close()
                        
                        


# In[ ]:


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
                        filename = "protein_node2vec_full_embeddings_"+str(EMBEDDING_DIM)+"_"+str(WALK_LENGTH)+"_"+str(NUM_WALK)+"_"+str(P)+"_"+str(Q)+".emb"
                        print(filename)
                        if not Path(filename).is_file():
                            node2vec = Node2Vec(G, dimensions=EMBEDDING_DIM, walk_length=WALK_LENGTH, num_walks=NUM_WALK, workers=WORKER, p = P, q = Q, temp_folder="tmp/")
                            model = node2vec.fit(window=10, min_count=1, batch_words=4)
                            model.wv.save_word2vec_format(filename)
                        node_embeddings = np.loadtxt(filename,skiprows=1)
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
                        kf = KFold(n_splits=5)
                        for train_index, test_index in kf.split(X2):
                            X_train2, X_test2 = X2[train_index], X2[test_index]
                            y_train2, y_test2 = y2[train_index], y2[test_index]
                            #clf = MLPClassifier(verbose=1)
                            #clf.fit(X_train2,y_train2)
                            clf = RandomForestClassifier(random_state=1,verbose=1,n_estimators=n_estimators_num)
                            clf.fit(X_train2,y_train2)

                            pred = clf.predict(X_test2)


                            roc.append(roc_auc_score(y_test2,pred, average='micro'))
                            prec.append(precision_score(y_test2,pred, average='micro'))
                            rec.append(recall_score(y_test2,pred, average='micro'))
                            f1.append(f1_score(y_test2,pred, average='micro'))

                        result = str(EMBEDDING_DIM)+","+str(WALK_LENGTH)+","+str(NUM_WALK)+","+str(P)+","+str(Q)+","+str(np.mean(roc))+","+str(np.mean(prec))+","+str(np.mean(rec))+","+str(np.mean(f1))+"\n"
                        f= open("result_pairwisenode_protein_node2vec.txt","a+")
                        f.write(result)
                        f.close()
                        
                        


# In[ ]:


EMBEDDING_DIMS = [64]
WALK_LENGTHS = [40]
NUM_WALKS = [50,100]
WORKERS = [4]
Ps = [1]
Qs = [1]
for EMBEDDING_DIM in EMBEDDING_DIMS:
    for WALK_LENGTH in WALK_LENGTHS:
        for NUM_WALK in NUM_WALKS:
            for WORKER in WORKERS:
                for P in Ps:
                    for Q in Qs:
                        filename = "protein_node2vec_full_embeddings_"+str(EMBEDDING_DIM)+"_"+str(WALK_LENGTH)+"_"+str(NUM_WALK)+"_"+str(P)+"_"+str(Q)+".emb"
                        print(filename)
                        if not Path(filename).is_file():
                            node2vec = Node2Vec(G, dimensions=EMBEDDING_DIM, walk_length=WALK_LENGTH, num_walks=NUM_WALK, workers=WORKER, p = P, q = Q, temp_folder="tmp/")
                            model = node2vec.fit(window=10, min_count=1, batch_words=4)
                            model.wv.save_word2vec_format(filename)
                        node_embeddings = np.loadtxt(filename,skiprows=1)
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
                        kf = KFold(n_splits=5)
                        for train_index, test_index in kf.split(X2):
                            X_train2, X_test2 = X2[train_index], X2[test_index]
                            y_train2, y_test2 = y2[train_index], y2[test_index]
                            #clf = MLPClassifier(verbose=1)
                            #clf.fit(X_train2,y_train2)
                            clf = RandomForestClassifier(random_state=1,verbose=1,n_estimators=n_estimators_num)
                            clf.fit(X_train2,y_train2)

                            pred = clf.predict(X_test2)


                            roc.append(roc_auc_score(y_test2,pred, average='micro'))
                            prec.append(precision_score(y_test2,pred, average='micro'))
                            rec.append(recall_score(y_test2,pred, average='micro'))
                            f1.append(f1_score(y_test2,pred, average='micro'))

                        result = str(EMBEDDING_DIM)+","+str(WALK_LENGTH)+","+str(NUM_WALK)+","+str(P)+","+str(Q)+","+str(np.mean(roc))+","+str(np.mean(prec))+","+str(np.mean(rec))+","+str(np.mean(f1))+"\n"
                        f= open("result_pairwisenode_protein_node2vec.txt","a+")
                        f.write(result)
                        f.close()
                        
                        


# In[ ]:


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
                        filename = "protein_node2vec_full_embeddings_"+str(EMBEDDING_DIM)+"_"+str(WALK_LENGTH)+"_"+str(NUM_WALK)+"_"+str(P)+"_"+str(Q)+".emb"
                        print(filename)
                        if not Path(filename).is_file():
                            node2vec = Node2Vec(G, dimensions=EMBEDDING_DIM, walk_length=WALK_LENGTH, num_walks=NUM_WALK, workers=WORKER, p = P, q = Q, temp_folder="tmp/")
                            model = node2vec.fit(window=10, min_count=1, batch_words=4)
                            model.wv.save_word2vec_format(filename)
                        node_embeddings = np.loadtxt(filename,skiprows=1)
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
                        kf = KFold(n_splits=5)
                        for train_index, test_index in kf.split(X2):
                            X_train2, X_test2 = X2[train_index], X2[test_index]
                            y_train2, y_test2 = y2[train_index], y2[test_index]
                            #clf = MLPClassifier(verbose=1)
                            #clf.fit(X_train2,y_train2)
                            clf = RandomForestClassifier(random_state=1,verbose=1,n_estimators=n_estimators_num)
                            clf.fit(X_train2,y_train2)

                            pred = clf.predict(X_test2)


                            roc.append(roc_auc_score(y_test2,pred, average='micro'))
                            prec.append(precision_score(y_test2,pred, average='micro'))
                            rec.append(recall_score(y_test2,pred, average='micro'))
                            f1.append(f1_score(y_test2,pred, average='micro'))

                        result = str(EMBEDDING_DIM)+","+str(WALK_LENGTH)+","+str(NUM_WALK)+","+str(P)+","+str(Q)+","+str(np.mean(roc))+","+str(np.mean(prec))+","+str(np.mean(rec))+","+str(np.mean(f1))+"\n"
                        f= open("result_pairwisenode_protein_node2vec.txt","a+")
                        f.write(result)
                        f.close()
                        
                        


# In[ ]:


EMBEDDING_DIMS = [64]
WALK_LENGTHS = [10]
NUM_WALKS = [40]
WORKERS = [4]
Ps = [1]
Qs = [0.2, 2]
for EMBEDDING_DIM in EMBEDDING_DIMS:
    for WALK_LENGTH in WALK_LENGTHS:
        for NUM_WALK in NUM_WALKS:
            for WORKER in WORKERS:
                for P in Ps:
                    for Q in Qs:
                        filename = "protein_node2vec_full_embeddings_"+str(EMBEDDING_DIM)+"_"+str(WALK_LENGTH)+"_"+str(NUM_WALK)+"_"+str(P)+"_"+str(Q)+".emb"
                        print(filename)
                        if not Path(filename).is_file():
                            node2vec = Node2Vec(G, dimensions=EMBEDDING_DIM, walk_length=WALK_LENGTH, num_walks=NUM_WALK, workers=WORKER, p = P, q = Q, temp_folder="tmp/")
                            model = node2vec.fit(window=10, min_count=1, batch_words=4)
                            model.wv.save_word2vec_format(filename)
                        node_embeddings = np.loadtxt(filename,skiprows=1)
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
                        kf = KFold(n_splits=5)
                        for train_index, test_index in kf.split(X2):
                            X_train2, X_test2 = X2[train_index], X2[test_index]
                            y_train2, y_test2 = y2[train_index], y2[test_index]
                            #clf = MLPClassifier(verbose=1)
                            #clf.fit(X_train2,y_train2)
                            clf = RandomForestClassifier(random_state=1,verbose=1,n_estimators=n_estimators_num)
                            clf.fit(X_train2,y_train2)

                            pred = clf.predict(X_test2)


                            roc.append(roc_auc_score(y_test2,pred, average='micro'))
                            prec.append(precision_score(y_test2,pred, average='micro'))
                            rec.append(recall_score(y_test2,pred, average='micro'))
                            f1.append(f1_score(y_test2,pred, average='micro'))

                        result = str(EMBEDDING_DIM)+","+str(WALK_LENGTH)+","+str(NUM_WALK)+","+str(P)+","+str(Q)+","+str(np.mean(roc))+","+str(np.mean(prec))+","+str(np.mean(rec))+","+str(np.mean(f1))+"\n"
                        f= open("result_pairwisenode_protein_node2vec.txt","a+")
                        f.write(result)
                        f.close()
                        
                        

