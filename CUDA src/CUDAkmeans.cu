/**
 * Universitˆ degli Studi di Milano
 * GPU Computing
 *
 * Title: Parallel K-Means applied on RGB Images.
 * Created: JULY 13, 2014 11:33AM
 * Author: Andrea Toscano
 *
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include "timer.h"

// Number of threads
#define BLOCK_SIZE 16
#define GRID_SIZE 256

//Useful to read Error from CUDA Calls
#define CUDA_CALL(x) {if((x) != cudaSuccess){ \
  printf("CUDA error at %s:%d\n",__FILE__,__LINE__); \
  printf("  %s\n", cudaGetErrorString(cudaGetLastError())); \
  exit(EXIT_FAILURE);}}

// nCentroids and size on device
__constant__ int dev_nCentroids;
__constant__ int dev_size;

// global variables
int PALETTE_BYTES = 0; // nCentroids * sizeof(int)
int IMAGE_BYTES = 0;  // width * height * sizeof(int)

//**********************************
//R,G,B Centroid's triple on device
// nCentroids on GPU is HARDCODED remind to update it manually!
__constant__ int dev_RedCentroid[20];
__constant__ int dev_GreenCentroid[20];
__constant__ int dev_BlueCentroid[20];
//**********************************
/*
 *  get RGB values from Initial Centroids (RAW format)
 *
 */
bool loadPalette(char* filename, int nCentroids, int* redCentroid, int* greenCentroid, int*  blueCentroid) {

	FILE *imageFile;
	int length = 0;

	imageFile = fopen(filename,"r");
	if (imageFile == NULL) {
		return false;
	} else {
		for (int i = 0; i < nCentroids; i++) {

			// R,G,B Centroid triple, nCentroids long
			redCentroid[i] = fgetc(imageFile);
			greenCentroid[i] = fgetc(imageFile);
			blueCentroid[i] = fgetc(imageFile);
			printf("%d, %d, %d\n",redCentroid[i], greenCentroid[i], blueCentroid[i] );
			length++;
		}
		fclose(imageFile);
		printf("\n");
		//printf("Palette Length: %d\n", length);
		return true;
	}
}

/*
 *  Image file loader (RAW format)
 *  Each pixel is identified by a triple r, g, b
 *  These values are taken from a raw file and put into three monodimensional arrays whose length is width*height
 */
bool loadRawImage(char* filename, int* r, int* g, int* b, int size) {
	FILE *imageFile;
	imageFile = fopen(filename, "r");

	if (imageFile == NULL) {
		return false;
	} else {
		for (int i = 0; i < size; i++) {

			r[i] = fgetc(imageFile);
			g[i] = fgetc(imageFile);
			b[i] = fgetc(imageFile);
		}
		fclose(imageFile);

		/*for(int j = 0; j < h * w; j++) {
			printf("%d, %d, %d ", r[j], g[j], b[j]);
		}*/
		return true;
	}
}


/*
 *  Image file writer (RAW format)
 */
bool writeRawImage(char* filename, int* labelArray, int* redCentroid, int* greenCentroid, int* blueCentroid, int size){
	FILE *imageFile;
	imageFile = fopen(filename, "wb");

	if(imageFile == NULL) {
		return false;
	} else {
		for (int i = 0; i < size; i++) {
			fputc((char) redCentroid[labelArray[i]], imageFile);
			fputc((char) greenCentroid[labelArray[i]], imageFile);
			fputc((char) blueCentroid[labelArray[i]], imageFile);
		}
		fclose(imageFile);
		return true;
	}
}

/*******************************************************************/

/*
 *  Clears arrays before each kernel getClusterLabel iteration
 *
 */
__global__ void clearPaletteArrays(int *dev_sumRed,int *dev_sumGreen,int *dev_sumBlue, int* dev_pixelClusterCounter, int* dev_tempRedCentroid, int* dev_tempGreenCentroid, int* dev_tempBlueCentroid ) {

	// 1 block, 16x16 threads
	int threadID = threadIdx.x + threadIdx.y * blockDim.x;

	if(threadID < dev_nCentroids) {

		// nCentroids long
		dev_sumRed[threadID] = 0;
		dev_sumGreen[threadID] = 0;
		dev_sumBlue[threadID] = 0;
		dev_pixelClusterCounter[threadID] = 0;
		dev_tempRedCentroid[threadID] = 0;
		dev_tempGreenCentroid[threadID] = 0;
		dev_tempBlueCentroid[threadID] = 0;
	}
}// end clearPaletteArrays

/*
 *  Clear label array before each kernel getClusterLabel iteration
 */
__global__ void clearLabelArray(int *dev_labelArray){

	// Global thread index
	int threadID = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;

	// labelArray is "size" long
	if(threadID < dev_size) {
		dev_labelArray[threadID] = 0;
	}
}// end clearLabelArray


/*
 * Finds the minimum distance between each triple dev_Red[i] dev_Green[i] dev_Blue[i] and all centroids.
 * Then saves the equivalent centroid label in dev_labelArray.
 * labelArray is   "width*height" long, monodimensional array
 *
 * INPUT : pixel triple arrays dev_Red, dev_Green, dev_Blue. labelArray that will contains the label for each pixel triple
 */
__global__ void getClusterLabel(int *dev_Red,int *dev_Green,int *dev_Blue,int *dev_labelArray) {


	// Global thread index
	int threadID = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;

	//default min value of distance
	float min = 500.0, value;
	//will be label
	int index = 0;


	if(threadID < dev_size) {
		// Finding the nearest centroid to current triple identified by threadID thread
		for(int i = 0; i < dev_nCentroids; i++) {
			// Performing Euclidean distance, Saving current value
			value = sqrtf(powf((dev_Red[threadID]-dev_RedCentroid[i]),2.0) + powf((dev_Green[threadID]-dev_GreenCentroid[i]),2.0) + powf((dev_Blue[threadID]-dev_BlueCentroid[i]),2.0));

			if(value < min){
				// saving new nearest centroid
				min = value;
				// Updating his index
				index = i;
			}
		}// end for
		// Writing to global memory the index of the nearest centroid
		// for dev_Red[threadID], dev_Green[threadID], dev_Blue[threadID] pixel triple
		dev_labelArray[threadID] = index;

	}// end if
}// end getClusterLabel





/*
 *  Summing Red, Green, Blue values per cluster
 *  Counting how many pixels there are in each cluster
 *
 */

__global__ void sumCluster(int *dev_Red,int *dev_Green,int *dev_Blue,int *dev_sumRed,int *dev_sumGreen,int *dev_sumBlue,int *dev_labelArray,int *dev_pixelClusterCounter) {

	// Global thread index
	int threadID = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;


	if(threadID < dev_size) {
		int currentLabelArray = dev_labelArray[threadID];
		int currentRed = dev_Red[threadID];
		int currentGreen = dev_Green[threadID];
		int currentBlue = dev_Blue[threadID];
		// Writing to global memory needs a serialization. Many threads are writing into the same few locations
		atomicAdd(&dev_sumRed[currentLabelArray], currentRed);
		atomicAdd(&dev_sumGreen[currentLabelArray], currentGreen);
		atomicAdd(&dev_sumBlue[currentLabelArray], currentBlue);
		atomicAdd(&dev_pixelClusterCounter[currentLabelArray], 1);
	}
}// end sumCluster


/*
 *  Calculates the new R,G,B values of the centroids dividing the sum of color (for each channel) by the number of pixels in that cluster
 *	New values are stored in global memory since the current R,G,B values of the centroids are in read-only constant memory.
 */
__global__ void newCentroids(int *dev_tempRedCentroid, int *dev_tempGreenCentroid, int *dev_tempBlueCentroid,int* dev_sumRed, int *dev_sumGreen,int *dev_sumBlue, int* dev_pixelClusterCounter) {

	// 1 block , 16*16 threads
	int threadID = threadIdx.x + threadIdx.y * blockDim.x;

	if(threadID < dev_nCentroids) {
		int currentPixelCounter = dev_pixelClusterCounter[threadID];
		int sumRed = dev_sumRed[threadID];
		int sumGreen = dev_sumGreen[threadID];
		int sumBlue = dev_sumBlue[threadID];

		//new RGB Centroids' values written in global memory
		dev_tempRedCentroid[threadID] = (int)(sumRed/currentPixelCounter);
		dev_tempGreenCentroid[threadID] = (int)(sumGreen/currentPixelCounter);
		dev_tempBlueCentroid[threadID] = (int)(sumBlue/currentPixelCounter);
	}

}// end newCentroids

/*******************************************************************/

int main(int argc, char *argv[]) {

		// init device
		cudaSetDevice(0);
		cudaDeviceSynchronize();
		cudaThreadSynchronize();

		//input raw file, output raw file, input palette raw file containing RGB values of initial centroids
		char *inputFile, *outputFile, *palette;
		//Pixels' r,g,b values. Centroid's r,g,b values
		int *red, *green, *blue, *redCentroid, *greenCentroid, *blueCentroid;

		// ref to GPU  Pixels'RGB values, Centroids' RGB values
		int *dev_Red, *dev_Green, *dev_Blue, *dev_tempRedCentroid, *dev_tempGreenCentroid, *dev_tempBlueCentroid;
		// array containing ref to GPU label array variable
		int *labelArray, *dev_labelArray;

		// local variables for storing image width, height
		// number of cluster, number of iterations, linear size of the image ( = width * height)
		int width, height, nCentroids, nIterations,size;
		//int IMAGE_BYTES, PALETTE_BYTES;

		// ref to array where pixels' count are stored
		int *pixelClusterCounter, *dev_pixelClusterCounter;
		// ref to arrays where sum of RGB values for each cluster are stored
		int *sumRed, *sumGreen, *sumBlue;
		int *dev_sumRed, *dev_sumGreen, *dev_sumBlue;

		// checking arguments
		if (argc > 7) {
			inputFile = argv[1];
			outputFile = argv[2];
			width = atoi(argv[3]);
			height = atoi(argv[4]);
			palette = argv[5];
			nCentroids = atoi(argv[6]);  // remind to update hardcoded nCentroids above
			if(nCentroids > 256)
				nCentroids = 256;
			nIterations = atoi(argv[7]);
			if(nIterations > 15)
				nIterations = 15;

		} else {
			printf("  USAGE: kmeans.cu <inputfile.raw> <outputfile.raw> nRows nCols palette nCentroids nItarations \n");
			printf("           <inputfile.raw>: input .raw file (sequence of bytes)\n");
			printf("          <outputfile.raw>: output .raw file\n");
			printf("                  nRows: the number of rows of the image\n");
			printf("                  nCols: the number of columns of the image\n");
			printf("			      palette: RGB initial Centroids");
			printf("			   nCentroids: number of clusters");
			printf("			  nIterations: number of iterations of K-Means");

			return 0;
		}

		// Setting image and palette size in bytes
		IMAGE_BYTES = width * height * sizeof(int);
		PALETTE_BYTES = nCentroids * sizeof(int);
		size = width * height;

		/* TODO useful for Texture implementation
		int ra[width][height];
		int ga[width][height];
		int ba[width][height];
		*/

		printf("Image: %s\n",inputFile);
		printf("Width: %d, Height: %d\n", width, height);
		printf("#Clusters: %d, #Iterations: %d\n", nCentroids, nIterations);


		// allocate memory on CPU
		red = static_cast<int *>(malloc(IMAGE_BYTES));
		green = static_cast<int *>(malloc(IMAGE_BYTES));
		blue = static_cast<int *>(malloc(IMAGE_BYTES));
		redCentroid = static_cast<int *>(malloc(PALETTE_BYTES));
		greenCentroid = static_cast<int *>(malloc(PALETTE_BYTES));
		blueCentroid = static_cast<int *>(malloc(PALETTE_BYTES));
		labelArray = static_cast<int *>(malloc(IMAGE_BYTES));
		sumRed = static_cast<int*>(malloc(PALETTE_BYTES));
		sumGreen = static_cast<int*>(malloc(PALETTE_BYTES));
		sumBlue = static_cast<int*>(malloc(PALETTE_BYTES));
		pixelClusterCounter = static_cast<int*>(malloc(PALETTE_BYTES));

		// Setting initial centroids
		printf("Initial Centroids: \n");
		if(loadPalette(palette, nCentroids, redCentroid, greenCentroid, blueCentroid)) {
		} else {
			printf("Unable to set Initial Centroids.\n");
		}

		// Loading image in r, g, b arrays
		printf("Image loading...\n");
		if (loadRawImage(inputFile, red, green, blue, size)) {
			printf("Image loaded!\n");
		} else {
			printf("NOT loaded!\n");
			return -1;
		}

		printf("\n");

		/* TODO Useful for Texture implementation
		int ri = 0;
		int co = 0;

		for( ri = 0; ri < width; ri++){
			for( co = 0; co < height; co++) {
			int x = ri+co*width;
			ra[ri][co] = red[x];
			ga[ri][co] = green[x];
			ba[ri][co] = blue[x];
			}
		}
*/

		if(IMAGE_BYTES == 0 || PALETTE_BYTES == 0) {
			return -1;
		}

		// allocate memory on GPU
		CUDA_CALL(cudaMalloc((void**) &dev_Red, IMAGE_BYTES));
		CUDA_CALL(cudaMalloc((void**) &dev_Green, IMAGE_BYTES));
		CUDA_CALL(cudaMalloc((void**) &dev_Blue, IMAGE_BYTES));
		CUDA_CALL(cudaMalloc((void**) &dev_tempRedCentroid, PALETTE_BYTES));
		CUDA_CALL(cudaMalloc((void**) &dev_tempGreenCentroid, PALETTE_BYTES));
		CUDA_CALL(cudaMalloc((void**) &dev_tempBlueCentroid, PALETTE_BYTES));
		CUDA_CALL(cudaMalloc((void**) &dev_labelArray, IMAGE_BYTES));
		CUDA_CALL(cudaMalloc((void**) &dev_sumRed, PALETTE_BYTES));
		CUDA_CALL(cudaMalloc((void**) &dev_sumGreen, PALETTE_BYTES));
		CUDA_CALL(cudaMalloc((void**) &dev_sumBlue, PALETTE_BYTES));
		CUDA_CALL(cudaMalloc((void**) &dev_pixelClusterCounter, PALETTE_BYTES));

		// copy host CPU memory to GPU
		CUDA_CALL(cudaMemcpy(dev_Red, red, IMAGE_BYTES, cudaMemcpyHostToDevice));
		CUDA_CALL(cudaMemcpy(dev_Green, green, IMAGE_BYTES, cudaMemcpyHostToDevice));
		CUDA_CALL(cudaMemcpy(dev_Blue, blue, IMAGE_BYTES, cudaMemcpyHostToDevice));
		CUDA_CALL(cudaMemcpy(dev_tempRedCentroid, redCentroid,PALETTE_BYTES,cudaMemcpyHostToDevice ));
		CUDA_CALL(cudaMemcpy(dev_tempGreenCentroid, greenCentroid,PALETTE_BYTES,cudaMemcpyHostToDevice ));
		CUDA_CALL(cudaMemcpy(dev_tempBlueCentroid, blueCentroid,PALETTE_BYTES,cudaMemcpyHostToDevice ));
		CUDA_CALL(cudaMemcpy(dev_labelArray, labelArray, IMAGE_BYTES, cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(dev_sumRed, sumRed, PALETTE_BYTES, cudaMemcpyHostToDevice));
		CUDA_CALL(cudaMemcpy(dev_sumGreen, sumGreen, PALETTE_BYTES, cudaMemcpyHostToDevice));
		CUDA_CALL(cudaMemcpy(dev_sumBlue, sumBlue, PALETTE_BYTES, cudaMemcpyHostToDevice));
		CUDA_CALL(cudaMemcpy(dev_pixelClusterCounter, pixelClusterCounter, PALETTE_BYTES, cudaMemcpyHostToDevice));
		CUDA_CALL(cudaMemcpyToSymbol(dev_RedCentroid, redCentroid, PALETTE_BYTES));
		CUDA_CALL(cudaMemcpyToSymbol(dev_GreenCentroid, greenCentroid, PALETTE_BYTES));
		CUDA_CALL(cudaMemcpyToSymbol(dev_BlueCentroid, blueCentroid, PALETTE_BYTES));
		CUDA_CALL(cudaMemcpyToSymbol(dev_nCentroids,&nCentroids, sizeof(int)));
		CUDA_CALL(cudaMemcpyToSymbol(dev_size, &size, sizeof(int)));


		// Clearing centroids on host
		for(int i = 0; i < nCentroids; i++) {
			redCentroid[i] = 0;
			greenCentroid[i] = 0;
			blueCentroid[i] = 0;
		}

		// Defining grid size

		int BLOCK_X, BLOCK_Y;
		BLOCK_X = ceil(width/BLOCK_SIZE);
		BLOCK_Y = ceil(height/BLOCK_SIZE);
		if(BLOCK_X > GRID_SIZE)
			BLOCK_X = GRID_SIZE;
		if(BLOCK_Y > GRID_SIZE)
			BLOCK_Y = GRID_SIZE;

		//2D Grid
		//Minimum number of threads that can handle width¡height pixels
	 	dim3 dimGRID(BLOCK_X,BLOCK_Y);
	 	//2D Block
	 	//Each dimension is fixed
		dim3 dimBLOCK(BLOCK_SIZE,BLOCK_SIZE);

		//Starting timer
		GpuTimer timer;
		timer.Start();

		printf("Launching K-Means Kernels..	\n");
		//Iteration of kmeans algorithm
		for(int i = 0; i < nIterations; i++) {


			// Passing image RGB components, palette RGB components, label Array, number of Clusters
			// Init  arrays' values to 0
			// Kernel needs only 1 block since nClusters
			clearPaletteArrays<<<1, dimBLOCK>>>(dev_sumRed, dev_sumGreen, dev_sumBlue, dev_pixelClusterCounter, dev_tempRedCentroid, dev_tempGreenCentroid, dev_tempBlueCentroid);

			// Init labelarray values to 0
			clearLabelArray<<<dimGRID, dimBLOCK>>>(dev_labelArray);

			// Calculates the distance from each pixel and all centroids
			// Then saves the equivalent label in dev_labelArray
			getClusterLabel<<< dimGRID, dimBLOCK >>> (dev_Red, dev_Green, dev_Blue,dev_labelArray);

			//Sums RGB values in each Cluster
			sumCluster<<<dimGRID, dimBLOCK>>> (dev_Red, dev_Green, dev_Blue, dev_sumRed, dev_sumGreen, dev_sumBlue, dev_labelArray,dev_pixelClusterCounter);

			//Finds new RGB Centroids' values
			newCentroids<<<1,dimBLOCK >>>(dev_tempRedCentroid, dev_tempGreenCentroid, dev_tempBlueCentroid, dev_sumRed, dev_sumGreen, dev_sumBlue, dev_pixelClusterCounter);

			//Old RGB Centroids' values are in constant memory
			//Updated RGB Centroids' values are in global memory
			//We need a swap
			CUDA_CALL(cudaMemcpy(redCentroid, dev_tempRedCentroid, PALETTE_BYTES,cudaMemcpyDeviceToHost));
			CUDA_CALL(cudaMemcpy(greenCentroid, dev_tempGreenCentroid, PALETTE_BYTES,cudaMemcpyDeviceToHost));
			CUDA_CALL(cudaMemcpy(blueCentroid, dev_tempBlueCentroid, PALETTE_BYTES,cudaMemcpyDeviceToHost));
			//Uploading in constant memory updated RGB Centroids' values
			CUDA_CALL(cudaMemcpyToSymbol(dev_RedCentroid, redCentroid, PALETTE_BYTES));
			CUDA_CALL(cudaMemcpyToSymbol(dev_GreenCentroid, greenCentroid, PALETTE_BYTES));
			CUDA_CALL(cudaMemcpyToSymbol(dev_BlueCentroid, blueCentroid, PALETTE_BYTES));
			timer.Stop();
		}

		// DEBUG
		CUDA_CALL(cudaMemcpy(labelArray, dev_labelArray, IMAGE_BYTES, cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(sumRed, dev_sumRed, PALETTE_BYTES, cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(sumGreen, dev_sumGreen, PALETTE_BYTES, cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(sumBlue, dev_sumBlue, PALETTE_BYTES, cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(pixelClusterCounter, dev_pixelClusterCounter, PALETTE_BYTES, cudaMemcpyDeviceToHost));

		printf("Kmeans code ran in: %f msecs.\n", timer.Elapsed());
		printf("\n");

		// labelArray DEBUG
		  int counter = 0;

		printf("Label Array:\n");
		for(int i = 0; i < (size); i++) {
			//printf("%d\n", labelArray[i]);
			counter++;
		}
		printf("printing counter %d\n", counter);
		counter = 0;

		printf("Sum Arrays:\n");
		for(int j = 0; j < nCentroids; j++) {
			printf("r: %u g: %u b: %u \n", sumRed[j], sumGreen[j], sumBlue[j]);
			counter++;
		}

		printf("\n");

		printf("Pixels per centroids:\n");
		for(int k = 0; k < nCentroids; k++){
			printf("%d centroid: %d pixels\n", k, pixelClusterCounter[k]);
		}

		printf("\n");



		printf("New centroids:\n");
		for(int i = 0; i < nCentroids; i++) {

			printf("%d, %d, %d \n", redCentroid[i], greenCentroid[i], blueCentroid[i]);
		}


		// writing...
		printf("Image writing...\n");

		if (writeRawImage(outputFile,labelArray, redCentroid, greenCentroid,  blueCentroid,  size)) {
			printf("Image written!\n");
		} else {
			printf("NOT written!\n");
			return -1;
		}

		free(red);
		free(green);
		free(blue);
		free(redCentroid);
		free(greenCentroid);
		free(blueCentroid);
		free(labelArray);
		free(sumRed);
		free(sumGreen);
		free(sumBlue);
		free(pixelClusterCounter);

		CUDA_CALL(cudaFree(dev_Red));
		CUDA_CALL(cudaFree(dev_Green));
		CUDA_CALL(cudaFree(dev_Blue));
		CUDA_CALL(cudaFree(dev_tempRedCentroid));
		CUDA_CALL(cudaFree(dev_tempGreenCentroid));
		CUDA_CALL(cudaFree(dev_tempBlueCentroid));
		CUDA_CALL(cudaFree(dev_labelArray));
		CUDA_CALL(cudaFree(dev_sumRed));
		CUDA_CALL(cudaFree(dev_sumGreen));
		CUDA_CALL(cudaFree(dev_sumBlue));
		CUDA_CALL(cudaFree(dev_pixelClusterCounter));

		printf("That's the end.\n");
		return 0;
}
