#include "../src/aes.h"
#include "stdio.h"
#include "../src/timer.h"



// Test function for keySchedule.
// Uses keys and values from the AES spec paper
void test_keySchedule() {
	unsigned char aeskey[16] = {0x2b ,0x7e ,0x15 ,0x16 ,0x28 ,0xae ,0xd2 ,0xa6 ,
								  0xab ,0xf7 ,0x15 ,0x88 ,0x09 ,0xcf ,0x4f ,0x3c};						  
	unsigned char expkey[11][16];
	
	keySchedule(aeskey, expkey);
	
	for (int c = 0; c < 11; c++) {
		for (int i = 0; i < 16; i++) {
			if (i%4==0) {
				printf("\nw%i : ",(c*16+i)/4);
			}
			printf("%x", expkey[c][i]);
		}
		printf("\n");
	}
}

void mixcols_test() {
	unsigned char ptxt[16] = {0x32 ,0x43 ,0xf6 ,0xa8 ,0x88 ,0x5a ,0x30 , 0x8d,
					0x31 ,0x31 ,0x98 ,0xa2 ,0xe0 ,0x37 ,0x07 ,0x34};
	
	//mixColumns(ptxt);
	//invMixColumns(ptxt);
	
	printf("Mixed columns:\n");
	for (int i = 0; i < 16; i++) {
		if (i%4==0) {
			printf("\nw%i : ",(i)/4);
		}
		printf("%02x", ptxt[i]);
	} 
	printf("\n");

}

void subbytes_test() {
	unsigned char ptxt[16] = {0x32 ,0x43 ,0xf6 ,0xa8 ,0x88 ,0x5a ,0x30 , 0x8d,
					0x31 ,0x31 ,0x98 ,0xa2 ,0xe0 ,0x37 ,0x07 ,0x34};
	
	//subBytes(ptxt);
	//invSubBytes(ptxt);
	
	for (int i = 0; i < 16; i++) {
		if (i%4==0) {
			printf("\nw%i : ",(i)/4);
		}
		printf("%02x", ptxt[i]);
	} 
	printf("\n");
}

void aes_test() {
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
	//aes128<<<dimGrid, dimBlock>>>(caeskey, cptxt);

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

}

void shiftRows_test() {
	unsigned char p[16] = {0x11,0x22,0x33,0x44, 0x1a,0x2a,0x3a,0x4a, 0x1b,0x2b,0x3b,0x4b, 0x1c,0x2c,0x3c,0x4c };
	
	printf("Original:\n");
	for (int i=0; i<16; i++) {
		printf("%2x,",p[i]);
	}
	printf("\n");
	
	//shiftRows(p);
	//invShiftRows(p);
	
	printf("Shifted:\n");
	for (int i=0; i<16; i++) {
		printf("%x,",p[i]);
	}
	printf("\n");
}


int main() {

	test_keySchedule();
}