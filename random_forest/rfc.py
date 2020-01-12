import numpy as np
import math
from sklearn.ensemble import RandomForestClassifier as rfc

def train_rfc(exp_max_features, min_leaf):
    train = np.load('train_vars.npy')
    val = np.load('val_vars.npy')
    train_labels = np.load('train_labels.npy').ravel()
    val_labels = np.load('val_labels.npy').ravel()
    val_size = len(val_labels)
    rfc_model = rfc(max_features = exp_max_features, min_samples_leaf = min_leaf)

    rfc_model.fit(train, train_labels)
    predictions = rfc_model.predict(val)
    

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
    return train_rfc(params['max_features'], params['leaf'])
