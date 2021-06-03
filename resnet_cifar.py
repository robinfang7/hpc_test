import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler
import time
import resnet
import argparse

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

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


(x,y), (x_test, y_test) = keras.datasets.cifar10.load_data()

train_dataset = tf.data.Dataset.from_tensor_slices((x,y))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

tf.random.set_seed(22)
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

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
file_writer.set_as_default()
tensorboard_callback = TensorBoard(
  log_dir=log_dir,
  update_freq='batch',
  histogram_freq=1)

lr_schedule_callback = LearningRateScheduler(schedule)

time_callback = TimeHistory()

model.fit(train_dataset,
          epochs=NUM_EPOCHS,
          validation_data=test_dataset,
          validation_freq=1,
          callbacks=[tensorboard_callback, lr_schedule_callback, time_callback])
model.evaluate(test_dataset)

#print(time_callback.times) # print each epoch's runtime
avg_time = sum(time_callback.times[1:])/len(time_callback.times[1:]) # remove first epoch
print('-'*40)
print("average of epoch time = %.2f " %(avg_time)) 
print("Throuthput = %.2f img/sec." % (NUM_TRAIN_IMG / avg_time))
print('-'*40) 

model.save('model.h5')

new_model = keras.models.load_model('model.h5')
 
new_model.evaluate(test_dataset)
