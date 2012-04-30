#ifndef SRC_AES_H
#define SRC_AES_H

// Header file

// AES core components


void keySchedule(unsigned char aeskey[16], unsigned char expandedkey[11][16]);

__device__
void addRoundKey(unsigned char block[16], unsigned char key[11][16], int round);

__device__
void subBytes(unsigned char block[16]);

__device__
void invSubBytes(unsigned char block[16]);

__device__
void shiftRows(unsigned char block[16]);

__device__
void invShiftRows(unsigned char block[16]);

__device__
void mixColumns(unsigned char block[16]);

__device__
void invMixColumns(unsigned char block[16]);

__global__
void aes128_core(unsigned char expandedkey[11][16], unsigned char *data);

void aes128(unsigned char key[16], unsigned char data[16]);

__global__
void invaes128_core(unsigned char expandedkey[11][16], unsigned char *data);

void invaes128(unsigned char key[16], unsigned char data[16]);


#endif
