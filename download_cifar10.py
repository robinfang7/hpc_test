import tensorflow as tf
from tensorflow.keras.datasets import cifar10
(x_img_train,y_label_train),(x_img_test,y_label_test)=cifar10.load_data()

print('train:',len(x_img_train))
print('test:',len(x_img_test))
