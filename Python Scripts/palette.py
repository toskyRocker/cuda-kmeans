'''
Andrea Toscano 
Universita' degli Studi di Milano
MS Computer Science
							   nameFile          n Clusters
USAGE :$   python palette.py photoName.jpg  		10    	    	
'''

# Import the modules
import sys, os, math, time
import numpy as np
from PIL import Image


# GLOBAL VARIABLES 
# Change these values if you want
MAX_CLUSTERS = 50

# Stores w, h of the image
IMAGE_SIZE = []
EXTENSION = ".jpg"



# Finding the starting colors (centroids)
def getPalette(nameFile):

	global IMAGE_SIZE 
	global CENTROIDS
	# Dim of each color palette
	dimEachColorPalette = 200
	image = 0
	try:
		image = Image.open(nameFile+EXTENSION)
	except IOError:
		print "IOError: Unable to open the image. "
		sys.exit()

	IMAGE_SIZE = image.size
	# Dim image
	print "Size Original Image: " + str(IMAGE_SIZE)
	# Dim of the resized image
	resize = int(0.5*IMAGE_SIZE[1])
    # Resize with antialias filter
	resizedImage = image.resize((resize, resize), Image.ANTIALIAS)
    # Get Palette
	result = resizedImage.convert('P', palette=Image.ADAPTIVE, colors=N_CLUSTERS)
	# Add Alpha channel -> PNG
	result.putalpha(0)
	# Get a list of tuple (count, color) 
	colors = result.getcolors(resize*resize)	
	colors = [col[1][:-1] for col in colors]
	
	array = np.asarray(colors, dtype=np.uint8)
	print "Printing Initial Centroids: "
	print array
	array.tofile(nameFile+"_palette"+str(N_CLUSTERS)+".raw")
	print nameFile+"_palette"+str(N_CLUSTERS)+".raw" + " saved."
	
	
# Main
if __name__ == "__main__":

    global N_CLUSTERS
	# Not considering .py namefile
    args = sys.argv[1:]
   	# Name input image 
    inputFile = args[0]
    print "File Name: " + str(inputFile)
    try:
        N_CLUSTERS = int(args[1])
    except: 
        print "Error: #Initial Colors > 1"
        sys.exit()

    print "#Initial Centroids: " + str(N_CLUSTERS) 

    nameFile,extension = os.path.splitext(inputFile)
    if (extension.lower() == ".jpg" or extension.lower() == ".jpeg") and N_CLUSTERS > 1 and N_CLUSTERS <= MAX_CLUSTERS:
    	getPalette(nameFile)
    else:
		print "Unable to load the image, probably your parameters are incorrect"
		print "USAGE:  palette.py photoName.jpg  		n    	 " 
		print "palette.py: name of this script "
		print "photoName.jpg: name of jpg input image"
		print "n: number of Clusters"
		sys.exit()

