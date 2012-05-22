#ifndef SRC_AES_H
#define SRC_AES_H

// Header file
// AES core components

#include "stdint.h"

typedef union {
	unsigned char b[4];
	uint32_t w;
} aesword_t;

void keySchedule(aesword_t aeskey[4], aesword_t expandedkey[11][4]);

__device__
void addRoundKey(aesword_t block[4], aesword_t key[11][4], int round);

__device__
void subBytes(aesword_t block[4]);

__device__
void invSubBytes(aesword_t block[4]);

__device__
void shiftRows(aesword_t block[4]); 

__device__
void invShiftRows(aesword_t block[4]);

__device__
void mixColumns(aesword_t block[4]);

__device__
void invMixColumns(aesword_t block[4]);

__global__ 
void aes128_ecb(aesword_t expandedkey[11][4], aesword_t *data);

__global__
void invaes128_ecb(aesword_t expandedkey[11][4], aesword_t *data);

__global__
void aes128_ctrc(aesword_t expandedkey[11][4], aesword_t *data, 
    aesword_t IV[4]);


#endif
