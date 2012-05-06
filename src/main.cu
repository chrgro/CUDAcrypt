#include "stdio.h"
#include "aes.h"
#include "timer.h"

__device__ unsigned char cexpkey[11][16];


const int THREADS_PER_BLOCK = 256;


// int main() {
	// aesword_t expkey[11][4];
	// aesword_t aeskey[4] = {0x2b ,0x7e ,0x15 ,0x16 ,0x28 ,0xae ,0xd2 ,0xa6 ,
						  // 0xab ,0xf7 ,0x15 ,0x88 ,0x09 ,0xcf ,0x4f ,0x3c};
						  
	// aesword_t plaintext[4] = {0x32 ,0x43 ,0xf6 ,0xa8 ,0x88 ,0x5a ,0x30 ,0x8d ,
			// 0x31 ,0x31 ,0x98 ,0xa2 ,0xe0 ,0x37 ,0x07 ,0x34};
			
	// keySchedule(aeskey, expkey);
	
	// aes128_core( expkey, plaintext);
	
	// printf("Encrypt:\n");
	// for (int c = 0; c < 4; c++) {
		// printf("w%i: ", c);
		// for (int r = 0; r < 4; r++) {
			// printf("%02x", plaintext[c].b[r]);
		// }
		// printf("\n");
	// }
	
	// invaes128_core( expkey, plaintext);
	// printf("Decrypt:\n");
	// for (int c = 0; c < 4; c++) {
		// printf("w%i: ", c);
		// for (int r = 0; r < 4; r++) {
			// printf("%02x", plaintext[c].b[r]);
		// }
		// printf("\n");
	// }

// }

int main(int argc, char *argv[]) {
	// Clear old error messages
	cudaGetLastError();

	/*Added by Richard for input output*/
	FILE *in_file, *out_file;
	int in_index = -1, out_index = -1; //the argument index corresponding to in/out
	const char* in_str = "-i";
	const char* out_str = "-o";
	
	for(int i = 0; i < argc-1; i++)
	{
		if(strcmp(argv[i], in_str) == 0)
		{
			in_index = i + 1;
			break;
		}
	}

	for(int i = 0; i < argc-1; i++)
	{
		if(strcmp(argv[i], out_str) == 0)
		{
			out_index = i + 1;
			break;
		}
	}
	
	if (in_index == -1) {
		printf("Incorrect input parameters!\nUsage: bin/cudacrypt -i <INPUTFILE> [-o <OUTPUTFILE>]\n");
		exit(-1);
	}
	
	in_file = fopen(argv[in_index], "rb");
	if (in_file == false) {
		printf("Error: Input file cannot be opened (check path)\n");
		exit(-1);
	}
	
	if (out_index != -1) {
		out_file = fopen(argv[out_index], "wb");	
		if (out_file == false) {
			printf("Error: Output file cannot be created \n");
			exit(-1);
		}
	}
	
	aesword_t *data;
	int datasize;
	int pad;
	
	float time;
	timerStart();
	fseek(in_file, 0L, SEEK_END);
	datasize = ftell(in_file);
	rewind (in_file);
	if(datasize%16) //not divisible by 16
		pad = 1;
	else//datasize is divisible by 16
		pad = 0;
	int numbytes = ((datasize/16) + pad) * 16;
	int numwords = numbytes/4;
	int numblocks = numwords/4;
	
	// Allocate pinned memory
	cudaHostAlloc( &data, numwords*sizeof(aesword_t), cudaHostAllocPortable);
	fread(data, 1, datasize, in_file);
	fclose(in_file);
	
	time = timerStop();
	printf("File-to-memory allocation: %fms\n", time);


	aesword_t expkey[11][4];
	aesword_t aeskey[4] = {0x2b ,0x7e ,0x15 ,0x16 ,0x28 ,0xae ,0xd2 ,0xa6 ,
						  0xab ,0xf7 ,0x15 ,0x88 ,0x09 ,0xcf ,0x4f ,0x3c};
						  
	// aesword_t plaintext[4] = {0x32 ,0x43 ,0xf6 ,0xa8 ,0x88 ,0x5a ,0x30 ,0x8d ,
			// 0x31 ,0x31 ,0x98 ,0xa2 ,0xe0 ,0x37 ,0x07 ,0x34};
						  

	printf("Incoming data:\n");
	//for (int i=0; i<4; i++) {printf("%08x", data[i].w); }
	for (int c = 0; c < 4; c++) {
		printf("w%i: ", c);
		for (int r = 0; r < 4; r++) {
			printf("%02x", data[c].b[r]);
		}
		printf("\n");
	}
	keySchedule(aeskey, expkey);
	// for (int c = 0; c < 11*4; c++) {
		// if (c%4==0 && c!=0) 
			// printf("\n");
		// printf("w%i: ", c);
		// for (int r = 0; r < 4; r++) {
			// printf("%02x", expkey[c/4][c%4].b[r]);
		// }
		// printf("\n");
	// }
	
	timerStart();
	
	// Set up GPU memory
	aesword_t *cdata;
	aesword_t *cexpkey;
	cudaMalloc ( &cdata, numbytes*sizeof(unsigned char));
	cudaMalloc ( &cexpkey, 11*16*sizeof(unsigned char));
	cudaMemcpy ( cdata, data, numbytes*sizeof(unsigned char), cudaMemcpyHostToDevice );
	cudaMemcpy ( cexpkey, expkey, 11*16*sizeof(unsigned char), cudaMemcpyHostToDevice );
	
	

	time = timerStop();
	printf ("Host-to-device data transfer: %fms\n", time);
	
	// Run
	dim3 dimGrid ( numblocks/THREADS_PER_BLOCK );
	dim3 dimBlock ( THREADS_PER_BLOCK );
	
	timerStart();
	if (dimGrid.x != 0) {
		aes128_core<<<dimGrid, dimBlock>>>((aesword_t(*)[4])cexpkey, cdata);
	}
	
	dim3 dimBlock_remaining ( numblocks % THREADS_PER_BLOCK );
	if (dimBlock_remaining.x != 0) {
		aes128_core<<<1, dimBlock_remaining>>>((aesword_t(*)[4])cexpkey, cdata+(dimGrid.x * THREADS_PER_BLOCK*4));
	} 
	
	time = timerStop();
	printf("Encryption time: %fms \n", time);
	
	printf("Blocks encrypted: %i\n", dimGrid.x*THREADS_PER_BLOCK + dimBlock_remaining.x);

	timerStart();
	cudaMemcpy ( data, cdata, numwords*sizeof(aesword_t), cudaMemcpyDeviceToHost );
	time = timerStop();
	printf("Device-to-host data transfer: %fms\n", time);
	
		
	printf("Encrypted data: \n");
	for (int c = 0; c < 4; c++) {
		if (c%4==0 && c!=0) 
			printf("\n");
		printf("w%i: ", c);
		for (int r = 0; r < 4; r++) {
			printf("%02x", data[c].b[r]);
		}
		printf("\n");
	}

	if (out_index != -1) {
		timerStart();
		fwrite (data , 1 , datasize , out_file);
		fclose (out_file);

		cudaFreeHost(data);
		time = timerStop();
		printf("Write to file and closing operations time: %f\n", time);
	} else {
		printf("No write to file requested.\n"); 
	}
	
	
	int err = cudaPeekAtLastError();
	if (err != 0) {
		printf("Some error occured with the CUDA device! Error code: %i\n", err);
	}
	
	return 0;
}