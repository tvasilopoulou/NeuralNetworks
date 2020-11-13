import numpy as np
import sys

def autoencoder(dataset):
	print(dataset)
	with open(dataset, "rb") as f:
		magicNum = int.from_bytes(f.read(4), byteorder = "big")
		numOfImages = int.from_bytes(f.read(4), byteorder = "big")
		dx = int.from_bytes(f.read(4), byteorder = "big")
		dy = int.from_bytes(f.read(4), byteorder = "big")
		
		print(str(magicNum))
		print(numOfImages)
		print(dx)
		print(dy)
		dimensions = dx*dy
		print(dimensions)
		byte = f.read(dimensions)
		while byte != b"" :
			byte = f.read(dimensions)


if __name__ == "__main__":
	if(len(sys.argv) != 3):
		sys.exit("Please try running autoencoder again. Number of arguments was different than expected.\n");
	autoencoder(sys.argv[2])