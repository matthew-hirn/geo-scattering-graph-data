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
