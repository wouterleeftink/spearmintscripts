import numpy as np
import math
from sklearn.svm import SVC as svm

def train_svm(experimental_c, experimental_gamma):
    train = np.load('train_vars.npy')
    val = np.load('val_vars.npy')
    train_labels = np.load('train_labels.npy').ravel()
    val_labels = np.load('val_labels.npy').ravel()
    val_size = len(val_labels)
    svm_model = svm(C = experimental_c, gamma = experimental_gamma)
    print 'Train y shape = %s' % (train_labels.shape,) 
    print 'Train X shape = %s' % (train.shape,)
    svm_model.fit(train, train_labels)
    predictions = svm_model.predict(val)
    
    print 'Val y shape = %s' % (val_labels.shape,)
    print 'Predictions shape = %s' % (predictions.shape,)
    correct = np.sum(np.equal(predictions, val_labels))
    accuracy = correct/float(val_size)
    result = 1-accuracy
    print 'Number of correct predictions: %f' % correct
    print 'Fraction of correct predictions: %f' % accuracy
    print 'Error rate: %f' % result
    print 'Number of labels %f' % len(val_labels)
    result = float(result)
    
    print 'Result = %f' % result
    #time.sleep(np.random.randint(60))
    return result

# Write a function like this called 'main'
def main(job_id, params):
    print 'Anything printed here will end up in the output directory for job #%d' % job_id
    print params
    return train_svm(params['c'], params['gamma'])
