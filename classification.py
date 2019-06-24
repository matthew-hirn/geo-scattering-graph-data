from geometric_scattering import *
from utilities import *
import scipy
import numpy as np

print('start reading file')
A, rX, Y = generate_graph(graph_name='dataset/imdb_comedy_romance_scifi.graph')
print('finish reading')

print('start feature generation')
feature = []

for i in range(len(A)):
    if i%200 == 0:
        print(i)
    #F = generate_mol_feature(A[i],t,maxvalue,rX[i])
    feature.append(generate_mol_feature(A[i],rX[i]))
print('finish feature generation')

feature = np.reshape(feature,(len(feature),feature[0].shape[0]))


print('feature shape',feature.shape)


#normalize feature
norm_feature = np.sum(np.power(feature,2),axis=0)

zero_list = []
for i in range(len(norm_feature)):
    if norm_feature[i] == 0:
        zero_list.append(i)



for i in reversed(zero_list):
    feature = np.concatenate((feature[:,0:i],feature[:,i+1:]),1)


print('feature shape',feature.shape)

feature_z = scipy.stats.mstats.zscore(feature,0)


print('feature z shape',feature_z.shape)

G_pool = [0.00001,0.0001,0.001, 0.01, 0.1, 1,10,100]
C_pool = [0.001, 0.01, 0.1, 1, 10,25,50,100,1000]

index = np.arange(len(rX))
np.random.shuffle(index)
#write_to_file(index)
n_splits = 10
shuffled_feature,shuffled_Y = shuffled(index,feature_z,Y)
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
