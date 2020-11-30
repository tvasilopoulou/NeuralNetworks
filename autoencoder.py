from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Dropout, Flatten
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
from keras.models import Model
from array import array
import numpy as np
import keras
import sys
 
def encoder(inputImg, layers, maxFilters = 64, convFiltSize = (3, 3)):
	flag = 2
	numOfFilters = (int)(maxFilters/(pow(2, ((layers-6)/2) + 1)))
	countLayers = 0
	conv = inputImg
	while countLayers < layers:
		conv = Conv2D(numOfFilters, convFiltSize, activation='relu', padding='same')(conv)
		numOfFilters *= 2
		conv = BatchNormalization()(conv)
		countLayers += 2
		if flag != 0:
			conv = MaxPooling2D(pool_size=(2, 2))(conv)
			# conv = Dropout(0.1)(conv)
			countLayers += 1
			flag -= 1
	return conv
 
def decoder(conv, layers, maxFilters, convFiltSize):
	countLayers = 0
	while countLayers < layers:
		conv = Conv2D(maxFilters, convFiltSize, activation='relu', padding='same')(conv)
		maxFilters /= 2
		conv = BatchNormalization()(conv)
		countLayers += 2
		if countLayers >= layers - 4: #6
			conv = UpSampling2D((2, 2))(conv)
			# conv = Dropout(0.1)(conv)
			countLayers += 1
	decoded = Conv2D(1, convFiltSize, activation='sigmoid', padding='same')(conv) # 28 x 28 x 1
	return decoded
 
def autoencoder(dataset, layers, maxFilters, x, y, convFiltSize, batchSize, epochs):
	images = []
	with open(dataset, "rb") as f:
		magicNum = int.from_bytes(f.read(4), byteorder = "big")
		numOfImages = int.from_bytes(f.read(4), byteorder = "big")
		dx = int.from_bytes(f.read(4), byteorder = "big")
		dy = int.from_bytes(f.read(4), byteorder = "big")
		
		dimensions = dx*dy
		# https://www.kaggle.com/hojjatk/read-mnist-dataset
		image_data = array("B", f.read()) 
 
	for i in range(numOfImages):
		images.append([0] * dx * dy)
	for i in range(numOfImages):
		img = np.array(image_data[i * dx * dy:(i + 1) * dx * dy])
		img = img.reshape(28, 28)
		images[i][:] = img
 

	
	inChannel = 1
	inputImage = Input(shape = (dx, dy, inChannel))

	autoencoder = Model(inputImage, decoder(encoder(inputImage, int(layers)/2, maxFilters, convFiltSize), int(layers)/2, maxFilters, convFiltSize))
	autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())
	# autoencoder.summary()
 
	xTrain, xValid, groundTrain, groundValid  = train_test_split(images, images, test_size=0.2, random_state=13)
	xTrain = np.array(xTrain).astype('float32') / 255.
	groundTrain = np.array(groundTrain).astype('float32') / 255.
	xValid = np.array(xValid).astype('float32') / 255.
	groundValid = np.array(groundValid).astype('float32') / 255.
	
 
	autoencoder_train = autoencoder.fit(xTrain, groundTrain, \
	 batch_size=batchSize,epochs=epochs,verbose=1,validation_data=(xValid, groundValid))

	lossFile = open("lossFile.txt", "a")
	lossFile.write(str(autoencoder_train.history['loss'][-1]) + "\n")
	lossFile.close()


	inp = input("We have produced the autoencoder model. Would you like to save? (Y/n) ")
	if inp == 'Y' or inp == 'y':
		path = input("Okay, please provide us with a path: ")
		# autoencoder.save_weights('./autoencoderWeights.h5')				#fix this
		autoencoder.save(path)
		print("Saved!")

	inp = input("Would you like to repeat your experiment with different hyperparameter values? (Y/n) ")
	if inp == 'Y' or inp == 'y':
		return 1
	inp = input("Would you like to plot your experiment's loss results? (Y/n) ")
	if inp == 'Y' or inp == 'y':
		loss = autoencoder_train.history['loss']
		val_loss = autoencoder_train.history['val_loss']
		epochs = range(epochs)
		plt.figure()
		plt.plot(epochs, loss, 'bo', label='Training loss')
		plt.plot(epochs, val_loss, label='Validation loss')
		plt.title('Training and validation loss')
		plt.legend()
		# plt.savefig('overfittingCheck.png')
		# plt.savefig('overfittingCheck.png', bbox_inches='tight')
		plt.show()
		return 0

	return 0

	# plt.savefig('./graphs/graph' + str(epochs[0]) +'.png')
 

if __name__ == "__main__":
	if(len(sys.argv) != 3):
		sys.exit("Please try running autoencoder again. Number of arguments was different than expected.\n");
	print("Welcome to Autoencoder. Before we get started, please provide us with a few parameter values. ")
	flag = 1
	while flag != 0:
		layers = input("Please enter number of layers: ") 
		maxFilters = input("Please enter number of filters (max): ") 
		x = input("Please enter a valid x dimension for the convolutional filters : ") 
		y = input("Please enter a valid y dimension for the convolutional filters : ") 
		convFiltSize = (int(x), int(y))
		batchSize = input("Please enter a batch size: ") 
		epochs = input("Please enter a number of epochs: ") 
		if (not(layers.isdigit())) or (not(maxFilters.isdigit())) or (not(x.isdigit())) or (not(y.isdigit())) or (not(batchSize.isdigit())) or (not(epochs.isdigit())):
			print("Something went wrong. Please try assigning integers as values.")
		else:
			promptStr = "Okay, so let's recap: you want " + str(layers) + " layers, " + str(maxFilters) + " maximum number of filters, convolutional filter size" + str(convFiltSize) + ", " + str(batchSize) + " sized batches and " + str(epochs) + " epoch(s). Correct? (Y/n) "
			answer = input(promptStr)
			if answer == 'Y' or answer == 'y':
				layers = int(layers)
				maxFilters = int(maxFilters)
				batchSize = int(batchSize)
				epochs = int(epochs)
				# break
			else:
				print("Okay let's try again.")
				continue

		flag = autoencoder(sys.argv[2], layers, maxFilters, x, y, convFiltSize, batchSize, epochs)
