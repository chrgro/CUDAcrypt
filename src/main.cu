#include "stdio.h"
#include "aes.h"
#include "timer.h"


__device__ unsigned char cexpkey[11][16];

int main(int argc, char *argv[]) {

	/*Added by Richard for input output*/
	FILE* in_file, out_file;
	int in_index, out_index; //the argument index corresponding to in/out
	const char* in_str = "-i";
	const char* out_str = "-c";
	for(int i = 0; i < argc; i++)
	{
		if(strcmp(argv[i], in_str) == 0)
		{
			in_index = i + 1;
			break;
		}
	}

	for(int i = 0; i < argc; i++)
	{
		if(strcmp(argv[i], out_str) == 0)
		{
			out_index = i + 1;
			break;
		}
	}

	in_file = fopen(argv[in_index], "rb");
	out_file = fopen(argv[out_index], "wb");	

	unsigned char *data;
	int datasize;
	int pad;

	fseek(in_file, 0L, SEEK_END);
	datasize = ftell(in_file);
	fseek(in_file, 0L, SEEK_SET);
	if(datasize%128) //not divisible by 128
		pad = 1;
	else//datasize is divisible by 128
		pad = 0;
	int numbytes = ((datasize/128) + pad) * 128;
	data = (unsigned char*)malloc(numbytes * sizeof(unsigned char));
	fread(data, 1, numbytes, in_file);


	unsigned char expkey[11][16];
	unsigned char aeskey[16] = {0x2b ,0x7e ,0x15 ,0x16 ,0x28 ,0xae ,0xd2 ,0xa6 ,
						  0xab ,0xf7 ,0x15 ,0x88 ,0x09 ,0xcf ,0x4f ,0x3c};
						  
	keySchedule(aeskey, expkey);
	
	
	float time;
	timerStart();
	
	// Set up GPU memory
	unsigned char *cdata;
	unsigned char cexpkey[11][16];
	cudaMalloc ( (void**)&cdata, numbytes*sizeof(unsigned char));
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
	fwrite (newdata , 1 , datasize*sizeof(unsigned char) , out_file);
	fclose (out_file);
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
