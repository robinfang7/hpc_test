import pickle

def unpickle(file):
    #import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

filename = "/home/u8880716/cifar-10-batches-py/data_batch_1" 
print(unpickle(filename).keys())
