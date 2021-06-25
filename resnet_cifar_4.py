import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler
import time
import resnet
import argparse
import pickle
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--num_gpus', type=int, default=2,
                    help='input gpu number, default=2')
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size, default=128')
parser.add_argument('--num_epochs', type=int, default=60,
                    help='input epoch, default=60')
args = parser.parse_args()

NUM_GPUS = args.num_gpus # 2 
BS_PER_GPU = args.batch_size # 128
NUM_EPOCHS = args.num_epochs # 60

HEIGHT = 32
WIDTH = 32
NUM_CHANNELS = 3
NUM_CLASSES = 10
NUM_TRAIN_SAMPLES = 50000
NUM_BATCHS = NUM_TRAIN_SAMPLES / (BS_PER_GPU * NUM_GPUS)
NUM_TRAIN_IMG = NUM_BATCHS * BS_PER_GPU * NUM_GPUS

BASE_LEARNING_RATE = 0.1
LR_SCHEDULE = [(0.1, 30), (0.01, 45)]

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

def normalize(x, y):
  x = tf.image.per_image_standardization(x)
  return x, y


def augmentation(x, y):
    x = tf.image.resize_with_crop_or_pad(
        x, HEIGHT + 8, WIDTH + 8)
    x = tf.image.random_crop(x, [HEIGHT, WIDTH, NUM_CHANNELS])
    x = tf.image.random_flip_left_right(x)
    return x, y	


def schedule(epoch):
  initial_learning_rate = BASE_LEARNING_RATE * BS_PER_GPU / 128
  learning_rate = initial_learning_rate
  for mult, start_epoch in LR_SCHEDULE:
    if epoch >= start_epoch:
      learning_rate = initial_learning_rate * mult
    else:
      break
  tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
  return learning_rate

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []
        self.batchtimes = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()
        self.batchtime = []

    def on_train_batch_begin(self, batch, log={}):
        self.batchtime_start = time.time()

    def on_train_batch_end(self, batch, log={}):
        self.batchtime.append(time.time() - self.batchtime_start)

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
        self.batchtimes.append(self.batchtime)

#(x,y), (x_test, y_test) = keras.datasets.cifar10.load_data()
#exit('stop')

# load data from folder /home/u8880716/cifar-10-batches-py
x = np.empty((0, 3, 32, 32), int)
y = np.empty((0,1), int)
for i in range(1,6):
    filename = '/home/u8880716/cifar-10-batches-py/data_batch_' + str(i)
    images = unpickle(filename)['data'] # tuple, ndarray (10000, 3072)
    labels = unpickle(filename)['labels'] # list
    imagearray = np.array(images).reshape(10000,3, 32,32)
    labelarray = np.array(labels).reshape(10000,1)
    x = np.append(x, imagearray, axis=0) # shape (50000,3,32,32)
    y = np.append(y, labelarray, axis=0)
    
filename = '/home/u8880716/cifar-10-batches-py/test_batch'
images = unpickle(filename)['data'] # tuple, ndarray (10000, 3072)
labels = unpickle(filename)['labels'] # list
x_test = np.array(images).reshape(10000,3, 32,32)
y_test = np.array(labels).reshape(10000,1)

x = x.transpose([0,2,3,1]) # convert dimension (3,32,32) to (32,32,3)
x_test = x_test.transpose([0,2,3,1])


train_dataset = tf.data.Dataset.from_tensor_slices((x,y))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

tf.random.set_seed(22)
#train_dataset = train_dataset.map(normalize).shuffle(NUM_TRAIN_SAMPLES).batch(BS_PER_GPU * NUM_GPUS, drop_remainder=True)
train_dataset = train_dataset.map(augmentation).map(normalize).shuffle(NUM_TRAIN_SAMPLES).batch(BS_PER_GPU * NUM_GPUS, drop_remainder=True)
test_dataset = test_dataset.map(normalize).batch(BS_PER_GPU * NUM_GPUS, drop_remainder=True)


input_shape = (HEIGHT, WIDTH, NUM_CHANNELS)
img_input = tf.keras.layers.Input(shape=input_shape)
opt = keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)

if NUM_GPUS == 1:
    model = resnet.resnet56(img_input=img_input, classes=NUM_CLASSES)
    model.compile(
              optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])
else:
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
      model = resnet.resnet56(img_input=img_input, classes=NUM_CLASSES)
      model.compile(
                optimizer=opt,
                loss='sparse_categorical_crossentropy',
                metrics=['sparse_categorical_accuracy'])  
'''
log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
file_writer.set_as_default()
tensorboard_callback = TensorBoard(
  log_dir=log_dir,
  update_freq='batch',
  histogram_freq=1)
'''
lr_schedule_callback = LearningRateScheduler(schedule)


time_callback = TimeHistory()

history = model.fit(train_dataset,
          epochs=NUM_EPOCHS,
          validation_data=test_dataset,
          validation_freq=1,
          callbacks=[lr_schedule_callback, time_callback])
model.evaluate(test_dataset)

avg_time = sum(time_callback.times[1:])/len(time_callback.times[1:]) # remove first epoch

# log accuracy and elasped time of epoch
logfile = "gpu" + str(NUM_GPUS) + \
          "_bs" + str(BS_PER_GPU) + \
          "_epoch" + str(NUM_EPOCHS) 

with open("%s.csv" % logfile, 'w') as f:
    f.write("loss,val_loss,accuracy,val_accuracy,epoch_elasped_time\n")
    for i in range(NUM_EPOCHS):
        f.write('%f,%f,%f,%f,%.2f \r\n' % \
               (history.history['loss'][i],\
                history.history['val_loss'][i],\
                history.history['sparse_categorical_accuracy'][i],\
                history.history['val_sparse_categorical_accuracy'][i],\
                time_callback.times[i]))

    f.write("average of epoch time = %.2f\n" %(avg_time))
    f.write("Throughput = %.2f img/sec.\n" % (NUM_TRAIN_IMG / avg_time))

#print("Epoch duration")
#print(time_callback.times) # print each epoch's runtime
#print("Batch duration of epoch")
#print(time_callback.batchtimes)

#print(sum(time_callback.times[1:]),len(time_callback.times[1:]))
#avg_time = sum(time_callback.times[1:])/len(time_callback.times[1:]) # remove first epoch
#print('-'*40)
#print("average of epoch time = %.2f " %(avg_time)) 
#print("Throughput = %.2f img/sec." % (NUM_TRAIN_IMG / avg_time))
#print('-'*40) 

#model.save('model.h5')

#new_model = keras.models.load_model('model.h5')
 
#new_model.evaluate(test_dataset)
