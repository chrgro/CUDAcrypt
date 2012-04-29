#include "stdio.h"
#include "aes.h"
#include "timer.h"


int main() {
	unsigned char aeskey[16] = {0x2b ,0x7e ,0x15 ,0x16 ,0x28 ,0xae ,0xd2 ,0xa6 ,
							  0xab ,0xf7 ,0x15 ,0x88 ,0x09 ,0xcf ,0x4f ,0x3c};		
	
	unsigned char ptxt[16] = {0x32 ,0x43 ,0xf6 ,0xa8 ,0x88 ,0x5a ,0x30 , 0x8d,
					0x31 ,0x31 ,0x98 ,0xa2 ,0xe0 ,0x37 ,0x07 ,0x34};

	printf("\nPlaintext:");
	for (int i = 0; i < 16; i++) {
		if (i%4==0) {
			printf("\nw%i : ",(i)/4);
		}
		printf("%02x", ptxt[i]);
	}
	printf("\n");	
	
	
	
	float time;
	timerStart();
	
	// Set up GPU memory
	unsigned char *cptxt;
	unsigned char *caeskey;
	cudaMalloc ( (void**)&cptxt, 16*sizeof(unsigned char));
	cudaMalloc ( (void**)&caeskey, 16*sizeof(unsigned char));
	cudaMemcpy ( cptxt, ptxt, 16*sizeof(unsigned char), cudaMemcpyHostToDevice );
	cudaMemcpy ( caeskey, aeskey, 16*sizeof(unsigned char), cudaMemcpyHostToDevice );

	time = timerStop();
	printf ("Elapsed memory transfer time: %fms\n", time);
	
	// Run
	dim3 dimBlock ( 1, 1 );
	dim3 dimGrid ( 1, 1 );
	
	timerStart();
	aes128<<<dimGrid, dimBlock>>>(caeskey, cptxt);

	time = timerStop();
	printf ("Elapsed action time: %fms\n", time);

	// Retrieve data
	timerStart();
	cudaMemcpy( ptxt, cptxt, 16*sizeof(unsigned char), cudaMemcpyDeviceToHost );
	cudaDeviceSynchronize();
	cudaFree( cptxt );

	time = timerStop();
	printf ("Elapsed memory writeback time: %fms\n", time);
	
	printf("\nCiphertext:");
	for (int i = 0; i < 16; i++) {
		if (i%4==0) {
			printf("\nw%i : ",(i)/4);
		}
		printf("%02x", ptxt[i]);
	} 
	printf("\n");
	
	return 0;
}

