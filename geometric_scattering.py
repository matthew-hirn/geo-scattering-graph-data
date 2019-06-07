import numpy as np
import networkx as nx
from numpy import linalg as LA


def lazy_random_walk(A):
    #if d == 0, P = 0
    P_array = []
    d = A.sum(0)
    P_t = A/d
    P_t[np.isnan(P_t)] = 0
    P = 1/2*(np.identity(P_t.shape[0])+P_t)
    #for i in range(0,t,2):
    #for i in range(t+1):
    #for i in [0]:
    #P_array.append(LA.matrix_power(P,i))
    #return P_array,P
    return P



def graph_wavelet(P):
    psi = []
    #for d1 in range(1,t,2):
    #for d1 in range(1,t+1):
    for d1 in [1,2,4,8,16]:
        W_d1 = LA.matrix_power(P,d1) - LA.matrix_power(P,2*d1)
        psi.append(W_d1)
    return psi



def zero_order_feature(ro,f):
    F0 = []
    for i in [1,2,3]:
        F0.append(np.sum(np.power(np.abs(ro),i),0))
    F0 = np.array(F0).reshape(-1,1)
    return F0



def first_order_feature(u,f):
    F1  = []
    for i in [1,2,3]:
        F1.append(np.sum(np.power(u,i),1))
    F1 = np.array(F1).reshape(-1,1)
    #F = []
    #F.append(np.sum(np.matmul(P,u[0]),1))
    #for i in range(1,t):
    #F = np.concatenate((F,np.sum(np.matmul(P,u[i]))),1)
    return F1



def selected_second_order_feature(W,u,f):
    u1 = np.einsum('ij,ajt ->ait',W[1],u[0:1])
    for i in range(2,len(W)):
        u1 = np.concatenate((u1,np.einsum('ij,ajt ->ait',W[i],u[0:i])),0)
    u1 = np.abs(u1)
    F2 = []
    for i in [1,2,3]:
        F2.append(np.sum(np.power(u1,i),1))
    F2 = np.array(F2).reshape(-1,1)
    #F2.append(np.sum(np.einsum('ij,ajt ->ait',W[i],u[0:i]),1).reshape(len(u[0:i])*F,1))
    #F2 = np.sum(np.einsum('ijk,akt ->iajt',P,F1),2).reshape(len(P)*len(F1)*F,1)
    #F2 = np.array(F2).reshape()
    return F2





def generate_mol_feature(A,f,ro):
    #with zero order, first order and second order features
    #shall consider only zero and first order features
    P = lazy_random_walk(A)
    W = graph_wavelet(P)
    u = np.abs(np.matmul(W,ro))
    
    F0 = zero_order_feature(ro,f)
    F1 = first_order_feature(u,f)
    #F2 = second_order_feature(W,u,P[0],t,F)
    F2 = selected_second_order_feature(W,u,f)
    #F3 = selected_third_order_feature(W,u,P[0],t,F)
    F = np.concatenate((F0,F1),axis=0)
    F = np.concatenate((F,F2),axis=0)
    #F = np.concatenate((F1,F2),axis=0)
    #F = np.concatenate((F,F3),axis=0)
    return F
