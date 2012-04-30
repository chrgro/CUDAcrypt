#include "stdio.h"
#include "aes.h"
#include "timer.h"


__device__ unsigned char cexpkey[11][16];

int main(int argc, char *argv[]) {

	const char *filename = "FIPS-197.pdf";
	
	// Open file in binary mode, find size
	FILE *fp = fopen(filename, "rb");
	
	
	
	unsigned char *data;
	int datasize;
	
	
	fseek(fp, 0L, SEEK_END);
	datasize = ftell(fp);
	fseek(fp, 0L, SEEK_SET);
	int numbytes = ((datasize/128)+1)*128;
	data = (unsigned char*) malloc(numbytes*sizeof(unsigned char));
	for (int i=numbytes-16; i<numbytes; i++) {
		data[i] = 0; // Clear the last 16 bytes
	}
	fread( data, 1, numbytes, fp); 
	
	

	printf("Datasize: %i, nearest bytes mod 128==0: %i\n", datasize, numbytes);
	printf("Number of blocks to be encrypted: %i\n", numbytes/128);
	
	
	
	
	unsigned char expkey[11][16];
	unsigned char aeskey[16] = {0x2b ,0x7e ,0x15 ,0x16 ,0x28 ,0xae ,0xd2 ,0xa6 ,
						  0xab ,0xf7 ,0x15 ,0x88 ,0x09 ,0xcf ,0x4f ,0x3c};
						  
	keySchedule(aeskey, expkey);
	
	
	float time;
	timerStart();
	
	// Set up GPU memory
	unsigned char *cdata;
	//unsigned char *ckey;
	cudaMalloc ( (void**)&cdata, numbytes*sizeof(unsigned char));
	//cudaMalloc ( (void**)&ckey, 16*sizeof(unsigned char));
	cudaMemcpy ( cdata, data, numbytes*sizeof(unsigned char), cudaMemcpyHostToDevice );
	cudaMemcpy ( cexpkey, expkey, 11*16*sizeof(unsigned char), cudaMemcpyHostToDevice );

	time = timerStop();
	printf ("Elapsed memory transfer time: %fms\n", time);
	
	// Run
	dim3 dimGrid ( (numbytes/128)/256 );
	dim3 dimBlock ( 256 );
	
	
	timerStart();
	aes128_core<<<dimGrid, dimBlock>>>(cexpkey, cdata);
	time = timerStop();
	printf("Encryption time: %fms \n", time);
	
	timerStart();
	unsigned char* newdata = (unsigned char*)malloc(numbytes*sizeof(unsigned char));
	cudaMemcpy ( newdata, cdata, numbytes*sizeof(unsigned char), cudaMemcpyDeviceToHost );
	FILE *pFile = fopen ( "encrFIPS-197.pdf" , "wb" );
	fwrite (newdata , 1 , datasize*sizeof(unsigned char) , pFile );
	fclose (pFile);
	time = timerStop();
	printf("Copy from device and to file: %fms\n", time);
	
	timerStart();
	fclose(fp);
	free(data);
	time = timerStop();
	printf("Closing operations time: %f\n", time);
	
	return 0;
}


void single_aes_example() {

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
