import numpy as np 
from unit import *
import keyboard
import time

def main():
	print("Building model..........................")
	Model = Model_sign_language()
	Model.load_weights('weight_model')
	print("Model load weight success !!!!")

	vid = cv2.VideoCapture(0)
	time_before = time.perf_counter()
	while True:
		ret,frame  = vid.read()
		cv2.imshow('frame',frame)
		time_now = time.perf_counter()
		if round(time_now - time_before,0) == 5.0:
			time_before = time_now
			num,percent = Model.predict_img(frame)
			print("Model predict number {0} with {1} %".format(num,percent*100))
		if cv2.waitKey(1) & 0xFF == ord('e'):
			break
	vid.release()
	cv2.destroyAllWindows()
	
if __name__ == "__main__":
	main()
