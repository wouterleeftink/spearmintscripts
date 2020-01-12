import numpy as np
import math
from sklearn.ensemble import AdaBoostClassifier as abc
from sklearn.tree import DecisionTreeClassifier

def train_abc(exp_depth, exp_lr):
    train = np.load('train_vars.npy')
    val = np.load('val_vars.npy')
    train_labels = np.load('train_labels.npy').ravel()
    val_labels = np.load('val_labels.npy').ravel()
    val_size = len(val_labels)
    abc_model = abc(base_estimator = DecisionTreeClassifier(max_depth=exp_depth), learning_rate=exp_lr)

    abc_model.fit(train, train_labels)
    predictions = abc_model.predict(val)
    

    correct = np.sum(np.equal(predictions, val_labels))
    accuracy = correct/float(val_size)
    result = 1-accuracy
    result = float(result)
    
    print 'Result = %f' % result
    #time.sleep(np.random.randint(60))
    return result

# Write a function like this called 'main'
def main(job_id, params):
    print 'Anything printed here will end up in the output directory for job #%d' % job_id
    print params
    return train_abc(params['depth'], params['lr'])
