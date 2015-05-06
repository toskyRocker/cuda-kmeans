'''
Andrea Toscano 
Universita' degli Studi di Milano
MS Computer Science
K-Means applied on RGB Images

With terminal move where your photos and kmeans.py are saved
Then execute kmeans.py script following this pattern. 

                              name of your photo  clusters no.     iterations no.
EXAMPLE :$   python kmeans.py photoName.jpg  		n    	    	m
'''

# Import the modules
import sys, os, math, time
from PIL import Image, ImageFilter, ImageDraw

####################
# GLOBAL VARIABLES #
####################
# Change these values if you want
MAX_CLUSTERS = 50
MAX_ITERATIONS = 50

# Stores w, h of the image
IMAGE_SIZE = []
EXTENSION = ".jpg"

# key: Cluster number, value: rgb value
CENTROIDS = {} 
NEW_CENTROIDS = {}

#list of [x,y,n_cluster] for each pixel
CLUSTER_NO = []

# Time
T0 = 0
T1 = 0

########################################
# Finding  starting colors (centroids) #
########################################
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
     
    # Save colors to file
	pal = Image.new('RGB', (dimEachColorPalette*numColors, dimEachColorPalette))
	draw = ImageDraw.Draw(pal)

	posx = 0
	arrayColors = []
	#print "Printing Initial Centroids: "
	for count, col in colors:
        #print count, col
        # Draw a rectangle with palette colors
		draw.rectangle([posx, 0, posx+dimEachColorPalette, dimEachColorPalette], fill=col)
        # List of the palette Colors without alpha channel aka Initial centers
		arrayColors.append(col[:-1])
		posx = posx + dimEachColorPalette

 	del draw
 	
	CENTROIDS = dict(zip(range(0, N_CLUSTERS), arrayColors))
	#print CENTROIDS
    
	pal.save(nameFile+"_cl"+str(N_CLUSTERS)+"_palette"+".png", "PNG")
	return image


########################################################################
# Euclidean distance between each pixel of the image and each centroid #
# Returns the number of centroid which is the nearest to a given pixel #
########################################################################
def findCluster(pixel):
	# Distance calculus
    dist = lambda pixel, clusterValue: math.sqrt( (clusterValue[1][0]-pixel[0])**2 + (clusterValue[1][1]-pixel[1])**2 + (clusterValue[1][2]-pixel[2])**2)
    # Vector with distances
    centroidDistances = ([dist(pixel, cv) for cv in CENTROIDS.items()])
    #print centroidDistances
    # Finding the nearest centroid and saving its number
    myCluster = centroidDistances.index(min([x for x in centroidDistances ]))
    #print myCluster
    return myCluster

def findNewMean(image):

	global NEW_CENTROIDS

	# Count how many pixel there are in each cluster
	count = [0] * N_CLUSTERS
	rgbValues = lambda: [0] * 3
	# Dictionary that saves the sum of r, g, b values of the pixel 
	rgb4EachCluster = {key : rgbValues() for key in range(N_CLUSTERS)}

	# For each triple [x,y,n_cluster]
	# Add +1 to the counter of the cluster where the pixel belongs to
	# Increment r,g,b values in the correct entry of the dict
	for triple in CLUSTER_NO:
		count[triple[2]] += 1
		rgb4EachCluster[triple[2]][0] += image[triple[0],triple[1]][0]
		rgb4EachCluster[triple[2]][1] += image[triple[0],triple[1]][1]
		rgb4EachCluster[triple[2]][2] += image[triple[0],triple[1]][2]

	#print "printing rgb4Cluster: " + str(rgb4EachCluster)
	#print "printing count: " + str(count)
	pos = 0

	# Finding the new centroids as 
	# sum(red_Color_of_total_pixels_in_a_given_centroid)/n_pixel_in_that_cluster
	# the same with g and b
	for pos in range(N_CLUSTERS):
		value = [0,0,0]
		if count[pos] == 0:
			value[0] = rgb4EachCluster[pos][0]
			value[1] = rgb4EachCluster[pos][1]
			value[2] = rgb4EachCluster[pos][2]
			#print rgb4EachCluster[pos][0],count[pos],value[0]
			#print rgb4EachCluster[pos][1],count[pos],value[1]
			#print rgb4EachCluster[pos][2],count[pos],value[2]
		else:
			value[0] = rgb4EachCluster[pos][0]/count[pos]
			value[1] = rgb4EachCluster[pos][1]/count[pos]
			value[2] = rgb4EachCluster[pos][2]/count[pos]
			#print rgb4EachCluster[pos][0],count[pos],value[0]
			#print rgb4EachCluster[pos][1],count[pos],value[1]
			#print rgb4EachCluster[pos][2],count[pos],value[2]
		NEW_CENTROIDS[pos] = value
	#print "Printing NEW CENTROIDS: " + str(NEW_CENTROIDS)

####################################
# Swaps NEW_CENTROIDS -> CENTROIDS #
####################################
def swapOldNewCentroids():
	global CENTROIDS
	tempDict = {k:v for k,v in CENTROIDS.items()}
	CENTROIDS = {}
	CENTROIDS = {k:v for k,v in NEW_CENTROIDS.items()}

###################################
# Saves the new clusterized image #
###################################
def saveNewImage(name):

	outputImage = Image.new("RGB",IMAGE_SIZE)
	print "Saving..."
	draw = ImageDraw.Draw(outputImage)

	for element in CLUSTER_NO:
		c = tuple(NEW_CENTROIDS.get(element[2]))
		draw.point((element[0],element[1]), fill=c)
	#outputImage.putdata(imageArray)
	outputImage.save(name+"_cl"+str(N_CLUSTERS)+"_it"+str(N_ITERATIONS)+EXTENSION,"JPEG",quality=100)
	print "That's the end"

########################
# Main k-means routine #
########################
def kmeans(image):
	
	global CLUSTER_NO
	global NEW_CENTROIDS
	# Loading the image
	loadedImage = image.load()
	#print loadedImage[0,0][0]
	x = 0
	y = 0
	# Finding initial classification
	for iter in range(N_ITERATIONS):
		print "#Iteration : " + str(iter)
		CLUSTER_NO = []
		NEW_CENTROIDS = {}

		# ClusterTuple has [x,y, n_cluster] for each pixel in the image
		# n_cluster is the number of cluster where the pixel belongs to
		for x in range(IMAGE_SIZE[0]):
			for y in range(IMAGE_SIZE[1]):
				clusterTuple = [0] * 3
				clusterTuple[0] = x # x position of the pixel
				clusterTuple[1] = y # y position of the pixel
				clusterTuple[2] = findCluster(loadedImage[x,y]) # cluster no. where the pixel belongs to
				CLUSTER_NO.append(clusterTuple)

		# Finds the new Cluster mean
		findNewMean(loadedImage)
		# Swaps the Dictionaries
		swapOldNewCentroids()
		
		#print "Printing CLUSTER NO " + str(CLUSTER_NO)
		#print "Printing NEW CENTROIDS: " + str(NEW_CENTROIDS)
		#print "Printing OLD CENTROIDS: " + str(CENTROIDS)
		#print ""
    #print "stamp Cluster no"
    #print CLUSTER_NO

##################
# Main procedure #
##################
def procedure(nameImage):

	# Finds initial centroids
	image = getPalette(nameImage)
	# Computes kmeans algorithm 
	# Taking K-Means time
	T0=time.time()
	kmeans(image)
	T1=time.time()
	print "Time: " + str(T1-T0) + " seconds."
	# Saves the new clusterized image
	saveNewImage(nameImage)

########
# Main #
########
if __name__ == "__main__":

    global N_ITERATIONS
    global N_CLUSTERS
	# Not considering .py namefile
    args = sys.argv[1:]
   	# Name input image 
    inputFile = args[0]
    print "File Name: " + str(inputFile)
    try:
        numColors = int(args[1])
        iterations = int(args[2])
        N_ITERATIONS = iterations  
        N_CLUSTERS = numColors
        
    except: 
        print "Error: #Colors or #Iteration are not correct"
        sys.exit()

    print "#Colors: " + str(numColors) + " #Iterations: " + str(iterations)

    nameFile,extension = os.path.splitext(inputFile)
    if (extension.lower() == ".jpg" or extension.lower() == ".jpeg") and numColors > 1 and numColors <= MAX_CLUSTERS and iterations > 0 and iterations <= MAX_ITERATIONS :
        procedure(nameFile)        
    else:
		print "Unable to load the image, probably your parameters are incorrect"
		sys.exit()