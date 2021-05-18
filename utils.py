import numpy as np 
import tensorflow as tf 
import cv2
from matplotlib import pyplot as plt

class Model_sign_language():
	def __init__(self):
		self.model = self.get_model()
	def get_model(self):
		input_shape = (64,64,3)
		input_img = tf.keras.Input(shape=input_shape)
		Z1 = tf.keras.layers.Conv2D(8,4,(1,1),padding='same')(input_img)
		A1 = tf.keras.layers.ReLU()(Z1)
		P1 = tf.keras.layers.MaxPool2D(pool_size=(8,8),strides=(8,8), padding='same')(A1)
		BN1 = tf.keras.layers.BatchNormalization(axis=-1)(P1)
		Z2 = tf.keras.layers.Conv2D(16,2,(1,1),padding='same')(BN1)
		A2 = tf.keras.layers.ReLU()(Z2)
		P2 = tf.keras.layers.MaxPool2D(pool_size=(4,4),strides=(4,4), padding='same')(A2)
		BN2 = tf.keras.layers.BatchNormalization(axis=-1)(P2)
		F = tf.keras.layers.Flatten()(BN2)
		outputs = tf.keras.layers.Dense(6,activation='softmax')(F)

		# YOUR CODE ENDS HERE
		model = tf.keras.Model(inputs=input_img, outputs=outputs)
		return model
	def load_weights(self,path_weight):
		self.model.load_weights(path_weight)
	def predict_img(self,frame):
		img_test_org = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
		img_test = cv2.resize(img_test_org,(64,64))
		cv2.imwrite('frame_test.jpg',img_test)
		img_test_fn = np.expand_dims(img_test,axis=0)
		result = self.model.predict(img_test_fn/255,batch_size=1)
		num_predict = np.argmax(result)
		percent = result.max()
		return num_predict,percent