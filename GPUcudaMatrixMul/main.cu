// Based nearly entirely on the code from the CUDA C Programming Guide

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "sys/time.h"


// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)

///////////////////////////////////////////////////////////////////////////////////////////////////
typedef struct {
int width;
int height;
float* elements;
int stride;
} Matrix;

// Thread block size
#define BLOCK_SIZE 16

//__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);
// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
///////////////////////////////////////////////////////////////////////////////////////////////////

struct tm *current;
time_t now;

//////////////////////////////////////////////////////////////////////////////
// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col) {
	return A.elements[row * A.stride + col];
}

//////////////////////////////////////////////////////////////////////////////
// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col, float value) {
	A.elements[row * A.stride + col] = value;
}


/////////////////////////////////////////////////////////////////////////////
// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
__device__ Matrix GetSubMatrix(Matrix A, int row, int col) {
	Matrix Asub;
	Asub.width = BLOCK_SIZE;
	Asub.height = BLOCK_SIZE;
	Asub.stride = A.stride;
	Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
	return Asub;
}





///////////////////////////////////////////////////////////////////////////
// Matrix multiplication kernel called by MatMul()

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
	// Block row and column

	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;
	// Each thread block computes one sub-matrix Csub of C
	Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
	// Each thread computes one element of Csub
	// by accumulating results into Cvalue
	float Cvalue = 0.0;
	// Thread row and column within Csub
	int row = threadIdx.y;
	int col = threadIdx.x;
	// Loop over all the sub-matrices of A and B that are
	// required to compute Csub
	// Multiply each pair of sub-matrices together
	// and accumulate the results
	for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
		// Get sub-matrix Asub of A
		Matrix Asub = GetSubMatrix(A, blockRow, m);
		// Get sub-matrix Bsub of B

		Matrix Bsub = GetSubMatrix(B, m, blockCol);
		// Shared memory used to store Asub and Bsub respectively
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
		// Load Asub and Bsub from device memory to shared memory
		// Each thread loads one element of each sub-matrix
		As[row][col] = GetElement(Asub, row, col);
		Bs[row][col] = GetElement(Bsub, row, col);
		// Synchronize to make sure the sub-matrices are loaded
		// before starting the computation
		__syncthreads();

		// Multiply Asub and Bsub together
		for (int e = 0; e < BLOCK_SIZE; ++e)
		{
			//this is the matrix multiplication statement
			//Cvalue = Cvalue + As[row][e] * Bs[e][col];
			//Cvalue = Cvalue + As[row][e] * Bs[e][col];


			//distance metric (not squared) Ricardo added
			Cvalue = Cvalue + ((As[row][e] - Bs[e][col]) * (As[row][e] - Bs[e][col]));
		}
		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
		__syncthreads();
	}
	// Write Csub to device memory
	// Each thread writes one element
	//printf("%.1f ", Cvalue);
	SetElement(Csub, row, col, Cvalue);
}



void MatMul(const Matrix A, const Matrix B, Matrix C) {
	// Load A and B to device memory

	Matrix d_A;
	d_A.width = d_A.stride = A.width;
	d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(float);
	cudaError_t err = cudaMalloc(&d_A.elements, size);
	printf("CUDA malloc A: %s\n",cudaGetErrorString(err));
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

	Matrix d_B;
	d_B.width = d_B.stride = B.width;
	d_B.height = B.height;
	size = B.width * B.height * sizeof(float);
	err = cudaMalloc(&d_B.elements, size);
	printf("CUDA malloc B: %s\n",cudaGetErrorString(err));
	cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

	// Allocate C in device memory
	Matrix d_C;
	d_C.width = d_C.stride = C.width;
	d_C.height = C.height;
	size = C.width * C.height * sizeof(float);
	err = cudaMalloc(&d_C.elements, size);
	printf("CUDA malloc C: %s\n",cudaGetErrorString(err));

	// Invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
	MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
	err = cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

	// Read C from device memory
	err = cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy C off of device: %s\n",cudaGetErrorString(err));

	// Free device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
}


//////////////////////////////////////////////
// Main()

int main(int argc, char* argv[]) {

	FILE *fp;
	FILE *fp_test;


	Matrix A, B, C;
	int a1, a2, b1, b2, i2;
	a1 = atoi(argv[1]); /* Height of A */
	a2 = atoi(argv[2]); /* Width of A */
	b1 = a2; /* Height of B */

	b2 = atoi(argv[3]); /* Width of B */

	//////////////////////////////////////////////////
	struct timeval detail_time;
	gettimeofday(&detail_time,NULL);
	printf("Begin milliseconds: %d, microseconds %d",
	detail_time.tv_usec /1000,  /* milliseconds */
	detail_time.tv_usec); /* microseconds */

	time(&now);
	current = localtime(&now);

	printf("\nGPU Calculation starts at time : %i:%i:%i\n", current->tm_hour, current->tm_min, current->tm_sec);

	///////////////////////////////////////////////////
	//float temp[a1][a2];

	int r = a1, c = a2, iii2;
	float *temp[r];
	for (iii2=0; iii2<r; iii2++)
			 temp[iii2] = (float *)malloc(c * sizeof(float));


    ////////////////////////////////////////////////////
	A.height = a1;
	A.width = a2;
	A.elements = (float*)malloc(A.width * A.height * sizeof(float));


	B.height = b1;
	B.width = b2;
	B.elements = (float*)malloc(B.width * B.height * sizeof(float));

	C.height = A.height; //a1 example 72
	C.width = B.width;  //b2 example 16
	C.elements = (float*)malloc(C.width * C.height * sizeof(float));


	/////////////////////////////////////////////////
/*
    //simple A= 16X32 and B=16X32
	if((fp = fopen("/home/purdueml/cuda-workspace/GPUmatrixMultiplication/Debug/gpudata.txt", "r")) == NULL)
	{
		printf("Error open input file!\n");
		exit(1);
	}

	if((fp_test = fopen("/home/purdueml/cuda-workspace/GPUmatrixMultiplication/Debug/gpudata_test.txt", "r")) == NULL)
	{
		printf("Error open input file!\n");
		exit(1);
	}
*/
    /////////////////////////////////////////////////////////////////////
/*
    //DARPA data A=72000X48 and B=16000X48
	if((fp = fopen("/home/purdueml/cuda-workspace/GPUmatrixMultiplication/Debug/data/TRAIN_GPU.72000x48_spaces.txt", "r")) == NULL)
	{
		printf("Error open input file!\n");
		exit(1);
	}

	if((fp_test = fopen("/home/purdueml/cuda-workspace/GPUmatrixMultiplication/Debug/data/TEST_GPU.16000x48_spaces.txt", "r")) == NULL)
	{
		printf("Error open input file!\n");
		exit(1);
	}
*/
	//////////////////////////////////////////////////////////////////////////
/*
	//darpa data - small data set A=16X48 and B=16X48
	if((fp = fopen("/home/purdueml/cuda-workspace/GPUmatrixMultiplication/Debug/data/bob_train.txt", "r")) == NULL)
	{
		printf("Error open input file!\n");
		exit(1);
	}

	if((fp_test = fopen("/home/purdueml/cuda-workspace/GPUmatrixMultiplication/Debug/data/bob_test.txt", "r")) == NULL)
	{
		printf("Error open input file!\n");
		exit(1);
	}
*/
	//////////////////////////////////////////////////////////////////////
/*
	//artificial data
	if((fp = fopen("/home/purdueml/cuda-workspace/GPUmatrixMultiplication/Debug/data/data/train80X48.txt", "r")) == NULL)
	{
		printf("Error open input file!\n");
		exit(1);
	}

	if((fp_test = fopen("/home/purdueml/cuda-workspace/GPUmatrixMultiplication/Debug/data/data/test16X48.txt", "r")) == NULL)
	{
		printf("Error open input file!\n");
		exit(1);
	}

	*/
    /////////////////////////////////////////////////////////////////////
	//artificial data
	if((fp = fopen("/home/purdueml/cuda-workspace/GPUmatrixMultiplication/Debug/data/data/iris_train160X16_gpu.csv", "r")) == NULL)
	{
		printf("Error open input file!\n");
		exit(1);
	}

	if((fp_test = fopen("/home/purdueml/cuda-workspace/GPUmatrixMultiplication/Debug/data/data/iris_test48X16_gpu.csv", "r")) == NULL)
	{
		printf("Error open input file!\n");
		exit(1);
	}

    /////////////////////////////////////////////////////////////////////


	i2 =0;
	while(fscanf(fp, "%f", &A.elements[i2])!= EOF) { i2 = i2 + 1; }
/*
    /////////////////////////////////////////////////////////////////////
	for(int ix = 0; ix < A.height; ix++)
	{
		for(int iy = 0; iy < A.width; iy++)
		{
	      fscanf(fp_test, "%f", &temp[ix][iy]);

		}
	}
*/
	for (int i = 0; i <  r; i++)
	{
	  for (int j = 0; j < c; j++)
	     {
		    fscanf(fp_test, "%f", &temp[i][j]);

		    //arr[i][j] = 2; // Or *(*(arr+i)+j) = ++count
		 }
    }
	////////////////////////////////////////////////////////////////////



/*
    // does nothing
	for(int i = 0; i < A.height; i++)
		for(int j = 0; j < A.width; j++)
		{
			//A.elements[i*A.width + j] = (arc4random() % 3);
			//A.elements[i*A.width + j] = (rand() % 13) + 1;  //use this one
			//A.elements[i*A.width + j] = A.elements[i*A.width + j] ;
		}
*/


	for(int i = 0; i < B.height; i++)
		for(int j = 0; j < B.width; j++)
		{
			//B.elements[i*B.width + j] = (arc4random() % 2);
			//B.elements[i*B.width + j] = (rand() % 12) + 1;
			B.elements[i*B.width + j] = temp[j][i];
		}

	/////////////////////////////////////////////////////////////////////////////////
	time(&now);
	current = localtime(&now);

	printf("\ntime just before GPU function call : %i:%i:%i\n", current->tm_hour, current->tm_min, current->tm_sec);



	//////////////////////////////////////////////////////////////////////////////////

	MatMul(A, B, C);
    //////////////////////////////////////////////////////////////////////////
	//print matrices A and B


	for(int i = 0; i < min(1000, A.height); i++){
		for(int j = 0; j < min(50, A.width); j++)
			printf("%.1f ", A.elements[i*A.width + j]);
		printf("\n");
	}

	printf("\n");

	for(int i = 0; i < min(1000, B.height); i++){
		for(int j = 0; j < min(50, B.width); j++)
			printf("%.1f ", B.elements[i*B.width + j]);
		printf("\n");
	}
	printf("\n");



	/////////////////////////////////////////////////////////////////////////
	gettimeofday(&detail_time,NULL);
	printf("End milliseconds: %d, microseconds %d",
	detail_time.tv_usec /1000,  /* milliseconds */
	detail_time.tv_usec); /* microseconds */


	time(&now);
	current = localtime(&now);
	printf("\nGPU calculation ends and serial lowest class search begins at time : %i:%i:%i\n", current->tm_hour, current->tm_min, current->tm_sec);

	///////////////////////////////////////////////////////////////////////


	///////////////////////////////////////////////////////////////////////
	// sorting may be needed here (parallel?) for more efficiency
	// not done





	///////////////////////////////////////////////////////////////////////
	//C.height = A.height; //a1 example 72
	//C.width = B.width;  //b2 example 16
	// select shortest distance
	float lowest = 999999;
	int index_of_closest_training_sample = -1; //-1 means nothing found
	for(int col = 0; col < C.width; col++)
	//for(int col = 0; col < 1; col++)
	{
			for(int row = 0; row <  C.height; row++)
			{
				if ((C.elements[col + row*C.width ]) < lowest){
					lowest = C.elements[col + row*C.width ];
					index_of_closest_training_sample = row;
				    //printf("%.1f \n", C.elements[col + row*C.width ]);
				}
			}
			//print nearest neighbor per test sample
			printf("test sample id:%d, closest_training_sample: %d, %.1f \n", col, index_of_closest_training_sample, lowest );
			lowest = 999999;
	}

	/////////////////////////////////////////////////////////////////////////
	gettimeofday(&detail_time,NULL);
	printf("End milliseconds: %d, microseconds %d",
	detail_time.tv_usec /1000,  /* milliseconds */
	detail_time.tv_usec); /* microseconds */


	time(&now);
	current = localtime(&now);
	printf("\n serial lowest  calculation ends at time : %i:%i:%i\n", current->tm_hour, current->tm_min, current->tm_sec);

	///////////////////////////////////////////////////////////////////////


	////////////////////////////////////////////////////////////////////////
	// print distance matrix

	for(int i = 0; i < min(1000, C.height); i++){
		for(int j = 0; j < min(1000, C.width); j++)
			printf("%.1f ", C.elements[i*C.width + j]);
		printf("\n");
	}
	printf("\n");


	////////////////////////////////////////////////////////////////////
	fclose(fp);
	fclose(fp_test);
	free(C.elements);
	free(B.elements );
	free(A.elements);

	for (int del_i=0; del_i<r; del_i++)
	{
		free(temp[del_i]);
	}


}
