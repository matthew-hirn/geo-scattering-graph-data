import numpy as np
from sklearn.decomposition import PCA
import scipy.stats.mstats
from sklearn.svm import SVC
from collections import Counter
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
    feature.append(generate_mol_feature(A[i],maxvalue,rX[i]))
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



G_pool = [0.00001,0.0001,0.001, 0.01, 0.1]



C_pool = [0.01, 0.1, 1, 10,25,50,100,1000]


index = np.arange(len(rX))
np.random.shuffle(index)
#write_to_file(index)
n_splits = 10
shuffled_feature,shuffled_Y = shuffled(index,pca_feature,Y)
train_id,test_id = Kfold(len(shuffled_feature),n_splits)
outter_loop = 0
test_accuracy = []
for k in range(n_splits):
    
    print('begin cross validation')
    outter_loop = outter_loop +1
    print('outter loop',outter_loop,'start')
    
    train_all_feature = [shuffled_feature[i] for i in train_id[k]]
    train_all_Y = [shuffled_Y[i] for i in train_id[k]]
    test_feature = [shuffled_feature[i] for i in test_id[k]]
    test_Y = [shuffled_Y[i] for i in test_id[k]]
    
    result,prediction_acc = cross_validate(9,train_all_feature,train_all_Y,test_feature,test_Y,outter_loop,G_pool,C_pool)
    #run_train(session, train_all_feature, train_all_Y)
    print("Cross-validation result: %s" % result)
    print('prediction accuracy',prediction_acc)
    test_accuracy.append(prediction_acc)
    print('outter loop',outter_loop,'ends')
#print("Test accuracy: %f" % session.run(accuracy, feed_dict={fea: test_feature, clas: test_Y}))
print(test_accuracy)
print('mean accuracy is ', np.mean(test_accuracy))
print('std is', np.std(test_accuracy))



