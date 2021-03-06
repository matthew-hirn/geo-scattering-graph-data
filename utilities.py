import pickle as pk
import networkx as nx
import numpy as np
from sklearn.svm import SVC
from collections import Counter

def generate_graph(graph_name='dataset/imdb_comedy_romance_scifi.graph'):
    if graph_name == 'dataset/imdb_comedy_romance_scifi.graph':
        maxval = 3
        n_classes = 3
    with open('dataset/imdb_comedy_romance_scifi.graph','rb') as f:
        new_f = pk._Unpickler(f)
        new_f.encoding = 'latin1'
        raw = new_f.load()
        
        n_graphs = len(raw['graph'])
        
        graph_list = []
        
        A = []
        rX = []
        Y = []
        
        
        for i in range(n_graphs):
            if i%200 == 0:
                print(i)
            class_label = int(raw['labels'][i])
            Y.append(class_label)
            
            # create graph
            g = raw['graph'][i]
            n_nodes = len(g)
            
            x = np.zeros((n_nodes, maxval), dtype='float32')
            
            G = nx.Graph()
            
            for node, meta in g.items():
                G.add_node(node)
                for neighbor in meta['neighbors']:
                    G.add_edge(node,neighbor)
                    
            for j in range(n_nodes):
                x[j,0] = nx.eccentricity(G,j)
                x[j,1] = nx.degree(G,j)
                x[j,2] = nx.clustering(G,j)
                
            graph_list.append(G)
            
            A.append(nx.adjacency_matrix(G,np.arange(n_nodes)).todense())
            rX.append(x)
    return A,rX,Y


def parse_graph_data(graph_name='dataset/enzymes.graph'):
    if graph_name == 'nci1.graph':
        maxval = 37
        n_classes = 2
    elif graph_name == 'nci109.graph':
        maxval = 38
        n_classes = 2
    elif graph_name == 'mutag.graph':
        maxval = 7
        n_classes = 2
    elif graph_name == 'ptc.graph':
        maxval = 22
        n_classes = 2
    elif graph_name == 'dataset/enzymes.graph':
        maxval = 3
        n_classes = 6
    
    with open(graph_name,'rb') as f:
        new_f = pk._Unpickler(f)
        new_f.encoding = 'latin1'
        raw = new_f.load()
        
        n_graphs = len(raw['graph'])
        
        A = []
        rX = []
        Y = []
        
        for i in range(n_graphs):
            # Set label
            class_label = raw['labels'][i]
            
            Y.append(class_label)
            
            # Parse graph
            G = raw['graph'][i]
            
            n_nodes = len(G)
            
            a = np.zeros((n_nodes, n_nodes), dtype='float32')
            x = np.zeros((n_nodes, maxval), dtype='float32')
            
            for node, meta in G.items():
                label = meta['label'][0] - 1
                x[node, label] = 1
                for neighbor in meta['neighbors']:
                    a[node, neighbor] = 1
            
            A.append(a)
            rX.append(x)

    return A, rX, Y


def cross_validate(split_size,train_all_feature,train_all_Y,test_feature,test_Y,outter_loop,G_pool,C_pool):
    results = []
    train_idx,val_idx = Kfold(len(train_all_feature),split_size)
    prediction = []
    
    
    test_feature = np.reshape(test_feature,(len(test_feature),len(test_feature[0])))
    
    
    for k in range(split_size):
        train_feature = [train_all_feature[i] for i in train_idx[k]]
        train_Y = [train_all_Y[i] for i in train_idx[k]]
        val_feature = [train_all_feature[i] for i in val_idx[k]]
        val_Y = [train_all_Y[i] for i in val_idx[k]]
        
        
        train_feature = np.reshape(train_feature,(len(train_feature),len(train_feature[0])))
        val_feature = np.reshape(val_feature,(len(val_feature),len(val_feature[0])))
        
    
        print('outter_loop',outter_loop)
        print('inner_loop',k,'start')
        
        print('start best para search')
        
        test_score,preds = run_train(train_feature,train_Y,G_pool,C_pool,val_feature,val_Y,test_feature,test_Y)
        #print('test score is', test_score)
        #print('start best para training/test')
        #score,preds = run_test(train_all_feature,train_all_Y,test_feature,test_Y,best_regu,best_W,best_epoch,learning_rate)
        #print('finished best epoch training')
        results.append(test_score)
        prediction.append(preds)
        #print(preds)
        print('this run accuracy is', results[-1])
        print('inner_loop',k,'ends')
    #print(prediction)
    #prediction = np.reshape(9,len(prediction[0]))
    
    prediction = np.array(prediction)
    #print(prediction.shape)
    pre = []
    for i in range(prediction.shape[1]):
        pre.append(Counter(prediction[:,i]).most_common(1)[0][0])
    test_acc = np.mean(np.equal(pre,test_Y))
    return (results,test_acc)


def run_train(train_feature,train_Y,G_pool,C_pool,val_feature,val_Y,test_feature,test_Y):
    temp = 0
    for c in C_pool:
        for g in G_pool:
            model = SVC(kernel='rbf',C=c,gamma=g)
            model.fit(train_feature,train_Y)
            score = model.score(val_feature,val_Y)
            if score >temp:
                temp =score
                test_score = model.score(test_feature,test_Y)
                preds = model.predict(test_feature)
    return (test_score,preds)


def shuffled(index,norm_feature,Y):
    new_feature = []
    new_Y = []
    for i in index:
        new_feature.append(norm_feature[i])
        new_Y.append(Y[i])
    return (new_feature,new_Y)



def Kfold(length,fold):
    size = np.arange(length).tolist()
    train_index = []
    val_index = []
    rest = length % fold
    fold_size = int(length/fold)
    temp_fold_size = fold_size
    for i in range(fold):
        temp_train = []
        temp_val = []
        if rest>0:
            temp_fold_size = fold_size+1
            rest = rest -1
            temp_val = size[i*temp_fold_size:+i*temp_fold_size+temp_fold_size]
            temp_train = size[0:i*temp_fold_size] + size[i*temp_fold_size+temp_fold_size:]
        else:
            temp_val = size[(length % fold)*temp_fold_size+(i-(length % fold))*fold_size
                            :(length % fold)*temp_fold_size+(i-(length % fold))*fold_size+fold_size]
            temp_train = size[0:(length % fold)*temp_fold_size+(i-(length % fold))*fold_size] + size[(length % fold)*temp_fold_size+(i-(length % fold))*fold_size+fold_size:]
        train_index.append(temp_train)
        val_index.append(temp_val)
    return (train_index,val_index)
