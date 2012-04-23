// This is the REAL "hello world" for CUDA!
// It takes the string "Hello ", prints it, then passes it to CUDA with an array
// of offsets. Then the offsets are added in parallel to produce the string "World!"
// By Ingemar Ragnemalm 2010

#include <stdio.h>

const int N = 16; 
const int blocksize = 16; 

__global__ 
void hello(char *a, int *b) 
{
        a[threadIdx.x] += b[threadIdx.x];
}


__global__
void otp(int *v, int *k) {

		v[threadIdx.x] = v[threadIdx.x] ^ (*k);
}


int main() {
		int vals[N] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
		int key = -1;
		
		for (int i=0; i<N; i++) {
			printf("%x,",vals[i]);
		}
		printf("\n");
		
		// Timer initiate
		cudaEvent_t start, stop; 
		float time; 
		cudaEventCreate(&start); 
		cudaEventCreate(&stop); 
		cudaEventRecord( start, 0 ); 
		
		
		// Set up GPU memory
		int *cvals;
		int *ckey;
		cudaMalloc ( (void**)&cvals, N*sizeof(int));
		cudaMalloc ( (void**)&ckey, sizeof(int));
		cudaMemcpy ( cvals, vals, N*sizeof(int), cudaMemcpyHostToDevice );
		cudaMemcpy ( ckey, &key, sizeof(int), cudaMemcpyHostToDevice );
		
		// Timer stop, reinitialize
		cudaEventRecord( stop, 0 ); 
		cudaEventSynchronize( stop ); 
		cudaEventElapsedTime( &time, start, stop ); 
		cudaEventDestroy( start ); 
		cudaEventDestroy( stop );
		cudaEventCreate(&start); 
		cudaEventCreate(&stop); 
		cudaEventRecord( start, 0 ); 
		
		// Output
		printf ("Elapsed memory transfer time: %fms\n", time);
		
		// Run OTP
		dim3 dimBlock ( blocksize, 1 );
		dim3 dimGrid ( 1, 1 );
		otp<<<dimGrid, dimBlock>>>(cvals, ckey);
		
		// Timer stop, reinitialize
		cudaEventRecord( stop, 0 ); 
		cudaEventSynchronize( stop ); 
		cudaEventElapsedTime( &time, start, stop ); 
		cudaEventDestroy( start ); 
		cudaEventDestroy( stop );
		cudaEventCreate(&start); 
		cudaEventCreate(&stop); 
		cudaEventRecord( start, 0 ); 
		
		// Output
		printf ("Elapsed OTP action time: %fms\n", time);
		
		// Retrieve data
		cudaMemcpy( vals, cvals, N*sizeof(int), cudaMemcpyDeviceToHost );
		cudaDeviceSynchronize();
		cudaFree( cvals );
		
		// Timer stop
		cudaEventRecord( stop, 0 ); 
		cudaEventSynchronize( stop ); 
		cudaEventElapsedTime( &time, start, stop ); 
		cudaEventDestroy( start ); 
		cudaEventDestroy( stop );
		
		// Output
		printf ("Elapsed memory writeback time: %fms\n", time);
		for (int i=0; i<N; i++) {
			printf("%x,",vals[i]);
		}
		printf("\n");

}

// Helloworld main
int hellomain()
{
        char a[N] = "Hello \0\0\0\0\0\0";
        int b[N] = {15, 10, 6, 0, -11, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        char *ad;
        int *bd;
        const int csize = N*sizeof(char);
        const int isize = N*sizeof(int);

        printf("%s", a);

        cudaMalloc( (void**)&ad, csize ); 
        cudaMalloc( (void**)&bd, isize ); 
        cudaMemcpy( ad, a, csize, cudaMemcpyHostToDevice ); 
        cudaMemcpy( bd, b, isize, cudaMemcpyHostToDevice ); 
        
        dim3 dimBlock( blocksize, 1 );
        dim3 dimGrid( 1, 1 );
        hello<<<dimGrid, dimBlock>>>(ad, bd);
        cudaMemcpy( a, ad, csize, cudaMemcpyDeviceToHost ); 
        cudaFree( ad );
        
        printf("%s\n", a);
        return EXIT_SUCCESS;
}