#include "stdio.h"
#include "aes.h"
#include "timer.h"


__device__ unsigned char cexpkey[11][16];

int main(int argc, char *argv[]) {
	// Clear old error messages
	cudaGetLastError();

	/*Added by Richard for input output*/
	FILE *in_file, *out_file;
	int in_index = 0, out_index = 0; //the argument index corresponding to in/out
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
	
	if (in_index == 0 || out_index == 0) {
		printf("Incorrect input parameters!\nUsage: bin/cudacrypt -i <INPUTFILE> -o <OUTPUTFILE>\n");
		exit(-1);
	}
	
	in_file = fopen(argv[in_index], "rb");
	if (in_file == false) {
		printf("Error: Input file cannot be opened (check path)\n");
		exit(-1);
	}
	
	out_file = fopen(argv[out_index], "wb");	
	if (out_file == false) {
		printf("Error: Output file cannot be created \n");
		exit(-1);
	}
	
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
	fclose(in_file);


	unsigned char expkey[11][16];
	unsigned char aeskey[16] = {0x2b ,0x7e ,0x15 ,0x16 ,0x28 ,0xae ,0xd2 ,0xa6 ,
						  0xab ,0xf7 ,0x15 ,0x88 ,0x09 ,0xcf ,0x4f ,0x3c};
						  
	keySchedule(aeskey, expkey);
	
	
	float time;
	timerStart();
	

	
	// Set up GPU memory
	unsigned char *cdata;
	unsigned char *cexpkey;
	cudaMalloc ( &cdata, numbytes*sizeof(unsigned char));
	cudaMalloc ( &cexpkey, 11*16*sizeof(unsigned char));
	cudaMemcpy ( cdata, data, numbytes*sizeof(unsigned char), cudaMemcpyHostToDevice );
	cudaMemcpy ( cexpkey, expkey, 11*16*sizeof(unsigned char), cudaMemcpyHostToDevice );
	
	

	time = timerStop();
	printf ("Elapsed memory transfer time: %fms\n", time);
	
	// Run
	dim3 dimGrid ( (numbytes/128)/256 );
	dim3 dimBlock ( 256 );
	
	
	timerStart();
	aes128_core<<<dimGrid, dimBlock>>>((unsigned char(*)[16])cexpkey, cdata);
	time = timerStop();
	printf("Encryption time: %fms \n", time);
	
	
	
	timerStart();
	unsigned char* newdata = (unsigned char*)malloc(numbytes*sizeof(unsigned char));
	cudaMemcpy ( newdata, cdata, numbytes*sizeof(unsigned char), cudaMemcpyDeviceToHost );
	time = timerStop();
	printf("Copy from device: %fms\n", time);
	
	
	timerStart();
	fwrite (newdata , 1 , datasize*sizeof(unsigned char) , out_file);
	fclose (out_file);

	free(data);
	time = timerStop();
	printf("Write to file and closing operations time: %f\n", time);
	
	int err = cudaPeekAtLastError();
	if (err != 0) {
		printf("Some error occured with the CUDA device! Error code: %i\n", err);
	}
	
	return 0;
}