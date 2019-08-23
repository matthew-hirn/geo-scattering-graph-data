import numpy as np
from sklearn.decomposition import PCA
import scipy.stats.mstats
from geometric_scattering import *
from utilities import *
import scipy

print('start reading file')
A, rX, Y = parse_graph_data(graph_name='dataset/enzymes.graph')
print('finish reading')

print('start feature generation')
feature = []

maxvalue = 3
for i in range(len(A)):
    if i%200 == 0:
        print(i)
    #F = generate_mol_feature(A[i],t,maxvalue,rX[i])
    feature.append(generate_mol_feature(A[i],rX[i]))
print('finish feature generation')

feature = np.reshape(feature,(len(feature),feature[0].shape[0]))


print('feature shape',feature.shape)


#remove 0 features
norm_feature = np.sum(np.power(feature,2),axis=0)



zero_list = []
for i in range(len(norm_feature)):
    if norm_feature[i] == 0.:
        zero_list.append(i)



for i in reversed(zero_list):
    feature = np.concatenate((feature[:,0:i],feature[:,i+1:]),1)


print('feature shape',feature.shape)


feature_z = scipy.stats.mstats.zscore(feature,0)



print('feature z shape',feature_z.shape)

#Let's first separate different classes.

class1 = []
class2 = []
class3 = []
class4 = []
class5 = []
class6 = []
for i in range(len(Y)):
    if Y[i] == 1:
        class1.append(feature_z[i])
    elif Y[i] == 2:
        class2.append(feature_z[i])
    elif Y[i] == 3:
        class3.append(feature_z[i])
    elif Y[i] == 4:
        class4.append(feature_z[i])
    elif Y[i] == 5:
        class5.append(feature_z[i])
    elif Y[i] == 6:
        class6.append(feature_z[i])

print('class1 size',len(class1))
print('class2 size',len(class2))
print('class3 size',len(class3))
print('class4 size',len(class4))
print('class5 size',len(class5))
print('class6 size',len(class6))

pca0 = PCA(n_components=0.9,svd_solver = 'full')
pca0.fit(feature_z)

pca_feature = pca0.fit_transform(feature_z)

print('pca all variance covered',pca0.explained_variance_ratio_)
print('pca all n components',pca0.n_components_)


pca1 = PCA(n_components=0.9,svd_solver = 'full')
pca1.fit(class1)
components1 = pca1.components_

print('pca class1 variance covered',pca1.explained_variance_ratio_)
print('pca class1 n components',pca1.n_components_)

pca2 = PCA(n_components=0.9,svd_solver = 'full')
pca2.fit(class2)
components2 = pca2.components_

print('pca class2 variance covered',pca2.explained_variance_ratio_)
print('pca class2 n components',pca2.n_components_) 

pca3 = PCA(n_components=0.9,svd_solver = 'full')
pca3.fit(class3)
components3 = pca3.components_
print('pca class3 variance covered',pca3.explained_variance_ratio_)
print('pca class3 n components',pca3.n_components_)

pca4 = PCA(n_components=0.9,svd_solver = 'full')
pca4.fit(class4)
components4 = pca4.components_
print('pca class4 variance covered',pca4.explained_variance_ratio_)
print('pca class4 n components',pca4.n_components_)

pca5 = PCA(n_components=0.9,svd_solver = 'full')
pca5.fit(class5)
components5 = pca5.components_
print('pca class5 variance covered',pca5.explained_variance_ratio_)
print('pca class5 n components',pca5.n_components_)

pca6 = PCA(n_components=0.9,svd_solver = 'full')
pca6.fit(class6)
components6 = pca6.components_
print('pca class6 variance covered',pca6.explained_variance_ratio_)
print('pca class6 n components',pca6.n_components_)

loss11 = 0
loss12 = 0
loss13 = 0
loss14 = 0
loss15 = 0
loss16 = 0

count1 = 0
count2 = 0
count3 = 0
count4 = 0
count5 = 0
count6 = 0

count1_ = 0
count2_ = 0
count3_ = 0
count4_ = 0
count5_ = 0
count6_ = 0

for i in range(len(class1)):
    Px1 = sum(np.multiply(np.matmul(components1,class1[i]),np.transpose(components1)).transpose())
    Px2 = sum(np.multiply(np.matmul(components2,class1[i]),np.transpose(components2)).transpose())
    Px3 = sum(np.multiply(np.matmul(components3,class1[i]),np.transpose(components3)).transpose())
    Px4 = sum(np.multiply(np.matmul(components4,class1[i]),np.transpose(components4)).transpose())
    Px5 = sum(np.multiply(np.matmul(components5,class1[i]),np.transpose(components5)).transpose())
    Px6 = sum(np.multiply(np.matmul(components6,class1[i]),np.transpose(components6)).transpose())
    
    temp1 = sum((class1[i]-Px1)**2)
    temp2 = sum((class1[i]-Px2)**2)
    temp3 = sum((class1[i]-Px3)**2)
    temp4 = sum((class1[i]-Px4)**2)
    temp5 = sum((class1[i]-Px5)**2)
    temp6 = sum((class1[i]-Px6)**2)
    
    pool = sorted([temp1,temp2,temp3,temp4,temp5,temp6])
    if pool[0] == temp1:
        count1 = count1+1
    if pool[1] == temp1:
        count1_ = count1_+1
    
    loss11 = temp1+loss11
    loss12 = temp2+loss12
    loss13 = temp3+loss13
    loss14 = temp4+loss14
    loss15 = temp5+loss15
    loss16 = temp6+loss16

dist11 = loss11/len(class1)
dist12 = loss12/len(class1)
dist13 = loss13/len(class1)
dist14 = loss14/len(class1)
dist15 = loss15/len(class1)
dist16 = loss16/len(class1)

print('dist11 is',dist11)
print('dist12 is',dist12)
print('dist13 is',dist13)
print('dist14 is',dist14)
print('dist15 is',dist15)
print('dist16 is',dist16)
print('count1 is',count1)
print('count1_ is',count1_)

loss21 = 0
loss22 = 0
loss23 = 0
loss24 = 0
loss25 = 0
loss26 = 0

for i in range(len(class2)):
    Px1 = sum(np.multiply(np.matmul(components1,class2[i]),np.transpose(components1)).transpose())
    Px2 = sum(np.multiply(np.matmul(components2,class2[i]),np.transpose(components2)).transpose())
    Px3 = sum(np.multiply(np.matmul(components3,class2[i]),np.transpose(components3)).transpose())
    Px4 = sum(np.multiply(np.matmul(components4,class2[i]),np.transpose(components4)).transpose())
    Px5 = sum(np.multiply(np.matmul(components5,class2[i]),np.transpose(components5)).transpose())
    Px6 = sum(np.multiply(np.matmul(components6,class2[i]),np.transpose(components6)).transpose())
    
    temp1 = sum((class2[i]-Px1)**2)
    temp2 = sum((class2[i]-Px2)**2)
    temp3 = sum((class2[i]-Px3)**2)
    temp4 = sum((class2[i]-Px4)**2)
    temp5 = sum((class2[i]-Px5)**2)
    temp6 = sum((class2[i]-Px6)**2)
    
    pool = sorted([temp1,temp2,temp3,temp4,temp5,temp6])
    if pool[0] == temp2:
        count2 = count2+1
    if pool[1] == temp2:
        count2_ = count2_+1
    
    loss21 = temp1+loss21
    loss22 = temp2+loss22
    loss23 = temp3+loss23
    loss24 = temp4+loss24
    loss25 = temp5+loss25
    loss26 = temp6+loss26

dist21 = loss21/len(class2)
dist22 = loss22/len(class2)
dist23 = loss23/len(class2)
dist24 = loss24/len(class2)
dist25 = loss25/len(class2)
dist26 = loss26/len(class2)

print('dist21 is',dist21)
print('dist22 is',dist22)
print('dist23 is',dist23)
print('dist24 is',dist24)
print('dist25 is',dist25)
print('dist26 is',dist26)
print('count2 is',count2)
print('count2_ is',count2_)

loss31 = 0
loss32 = 0
loss33 = 0
loss34 = 0
loss35 = 0
loss36 = 0

for i in range(len(class3)):
    Px1 = sum(np.multiply(np.matmul(components1,class3[i]),np.transpose(components1)).transpose())
    Px2 = sum(np.multiply(np.matmul(components2,class3[i]),np.transpose(components2)).transpose())
    Px3 = sum(np.multiply(np.matmul(components3,class3[i]),np.transpose(components3)).transpose())
    Px4 = sum(np.multiply(np.matmul(components4,class3[i]),np.transpose(components4)).transpose())
    Px5 = sum(np.multiply(np.matmul(components5,class3[i]),np.transpose(components5)).transpose())
    Px6 = sum(np.multiply(np.matmul(components6,class3[i]),np.transpose(components6)).transpose())
    
    temp1 = sum((class3[i]-Px1)**2)
    temp2 = sum((class3[i]-Px2)**2)
    temp3 = sum((class3[i]-Px3)**2)
    temp4 = sum((class3[i]-Px4)**2)
    temp5 = sum((class3[i]-Px5)**2)
    temp6 = sum((class3[i]-Px6)**2)
    
    pool = sorted([temp1,temp2,temp3,temp4,temp5,temp6])
    if pool[0] == temp3:
        count3 = count3+1
    if pool[1] == temp3:
        count3_ = count3_+1
    
    loss31 = temp1+loss31
    loss32 = temp2+loss32
    loss33 = temp3+loss33
    loss34 = temp4+loss34
    loss35 = temp5+loss35
    loss36 = temp6+loss36

dist31 = loss31/len(class3)
dist32 = loss32/len(class3)
dist33 = loss33/len(class3)
dist34 = loss34/len(class3)
dist35 = loss35/len(class3)
dist36 = loss36/len(class3)

print('dist31 is',dist31)
print('dist32 is',dist32)
print('dist33 is',dist33)
print('dist34 is',dist34)
print('dist35 is',dist35)
print('dist36 is',dist36)
print('count3 is',count3)
print('count3_ is',count3_)

loss41 = 0
loss42 = 0
loss43 = 0
loss44 = 0
loss45 = 0
loss46 = 0

for i in range(len(class4)):
    Px1 = sum(np.multiply(np.matmul(components1,class4[i]),np.transpose(components1)).transpose())
    Px2 = sum(np.multiply(np.matmul(components2,class4[i]),np.transpose(components2)).transpose())
    Px3 = sum(np.multiply(np.matmul(components3,class4[i]),np.transpose(components3)).transpose())
    Px4 = sum(np.multiply(np.matmul(components4,class4[i]),np.transpose(components4)).transpose())
    Px5 = sum(np.multiply(np.matmul(components5,class4[i]),np.transpose(components5)).transpose())
    Px6 = sum(np.multiply(np.matmul(components6,class4[i]),np.transpose(components6)).transpose())
    
    temp1 = sum((class4[i]-Px1)**2)
    temp2 = sum((class4[i]-Px2)**2)
    temp3 = sum((class4[i]-Px3)**2)
    temp4 = sum((class4[i]-Px4)**2)
    temp5 = sum((class4[i]-Px5)**2)
    temp6 = sum((class4[i]-Px6)**2)
    
    pool = sorted([temp1,temp2,temp3,temp4,temp5,temp6])
    if pool[0] == temp4:
        count4 = count4+1
    if pool[1] == temp4:
        count4_ = count4_+1
    
    loss41 = temp1+loss41
    loss42 = temp2+loss42
    loss43 = temp3+loss43
    loss44 = temp4+loss44
    loss45 = temp5+loss45
    loss46 = temp6+loss46

dist41 = loss41/len(class4)
dist42 = loss42/len(class4)
dist43 = loss43/len(class4)
dist44 = loss44/len(class4)
dist45 = loss45/len(class4)
dist46 = loss46/len(class4)

print('dist41 is',dist41)
print('dist42 is',dist42)
print('dist43 is',dist43)
print('dist44 is',dist44)
print('dist45 is',dist45)
print('dist46 is',dist46)
print('count4 is',count4)
print('count4_ is',count4_)

loss51 = 0
loss52 = 0
loss53 = 0
loss54 = 0
loss55 = 0
loss56 = 0

for i in range(len(class5)):
    Px1 = sum(np.multiply(np.matmul(components1,class5[i]),np.transpose(components1)).transpose())
    Px2 = sum(np.multiply(np.matmul(components2,class5[i]),np.transpose(components2)).transpose())
    Px3 = sum(np.multiply(np.matmul(components3,class5[i]),np.transpose(components3)).transpose())
    Px4 = sum(np.multiply(np.matmul(components4,class5[i]),np.transpose(components4)).transpose())
    Px5 = sum(np.multiply(np.matmul(components5,class5[i]),np.transpose(components5)).transpose())
    Px6 = sum(np.multiply(np.matmul(components6,class5[i]),np.transpose(components6)).transpose())
    
    temp1 = sum((class5[i]-Px1)**2)
    temp2 = sum((class5[i]-Px2)**2)
    temp3 = sum((class5[i]-Px3)**2)
    temp4 = sum((class5[i]-Px4)**2)
    temp5 = sum((class5[i]-Px5)**2)
    temp6 = sum((class5[i]-Px6)**2)
    
    pool = sorted([temp1,temp2,temp3,temp4,temp5,temp6])
    if pool[0] == temp5:
        count5 = count5+1
    if pool[1] == temp5:
        count5_ = count5_+1
    
    loss51 = temp1+loss51
    loss52 = temp2+loss52
    loss53 = temp3+loss53
    loss54 = temp4+loss54
    loss55 = temp5+loss55
    loss56 = temp6+loss56

dist51 = loss51/len(class5)
dist52 = loss52/len(class5)
dist53 = loss53/len(class5)
dist54 = loss54/len(class5)
dist55 = loss55/len(class5)
dist56 = loss56/len(class5)

print('dist51 is',dist51)
print('dist52 is',dist52)
print('dist53 is',dist53)
print('dist54 is',dist54)
print('dist55 is',dist55)
print('dist56 is',dist56)
print('count5 is',count5)
print('count5_ is',count5_)


loss61 = 0
loss62 = 0
loss63 = 0
loss64 = 0
loss65 = 0
loss66 = 0

for i in range(len(class6)):
    Px1 = sum(np.multiply(np.matmul(components1,class6[i]),np.transpose(components1)).transpose())
    Px2 = sum(np.multiply(np.matmul(components2,class6[i]),np.transpose(components2)).transpose())
    Px3 = sum(np.multiply(np.matmul(components3,class6[i]),np.transpose(components3)).transpose())
    Px4 = sum(np.multiply(np.matmul(components4,class6[i]),np.transpose(components4)).transpose())
    Px5 = sum(np.multiply(np.matmul(components5,class6[i]),np.transpose(components5)).transpose())
    Px6 = sum(np.multiply(np.matmul(components6,class6[i]),np.transpose(components6)).transpose())
    
    temp1 = sum((class6[i]-Px1)**2)
    temp2 = sum((class6[i]-Px2)**2)
    temp3 = sum((class6[i]-Px3)**2)
    temp4 = sum((class6[i]-Px4)**2)
    temp5 = sum((class6[i]-Px5)**2)
    temp6 = sum((class6[i]-Px6)**2)
    
    pool = sorted([temp1,temp2,temp3,temp4,temp5,temp6])
    if pool[0] == temp6:
        count6 = count6+1
    if pool[1] == temp6:
        count6_ = count6_+1
    
    loss61 = temp1+loss61
    loss62 = temp2+loss62
    loss63 = temp3+loss63
    loss64 = temp4+loss64
    loss65 = temp5+loss65
    loss66 = temp6+loss66

dist61 = loss61/len(class6)
dist62 = loss62/len(class6)
dist63 = loss63/len(class6)
dist64 = loss64/len(class6)
dist65 = loss65/len(class6)
dist66 = loss66/len(class6)

print('dist61 is',dist61)
print('dist62 is',dist62)
print('dist63 is',dist63)
print('dist64 is',dist64)
print('dist65 is',dist65)
print('dist66 is',dist66)
print('count6 is',count6)
print('count6_ is',count6_)
