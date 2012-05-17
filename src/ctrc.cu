/*
 * aes with 128 bit key in CTRC mode. IV is treated as an int of size 128, 
 * but given to us as an unsigned char array of size 16
 */

__global__
void aes128_ctrc(unsigned char expandedkey[11][16], unsigned char *data,
    unsigned char IV[16])
{
  //thread number #
  long tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned char ctr[16]; //this is IV + tid

  //this part does ctrc = IV + tid;

  for(int i = 0; i < 16; i++)
  {
    //since IV is at most 256, it will wrap around for each 256 
    int j = tid%256;
    //add most significant part first
    ctr[i] = IV[i] + j%(16 * (16 - i - 1));
    //remove that part you just added from tid
    j = j/16;
  }

  //now ctrc is set, and we need to encrypt it, start AES rounds

  //1. initial round
  int round = 0;
  addRoundKey(ctr, expandedkey, round);

  //2. repeat round 9 times
  for(round = 1; round < 10; round++)
  {
    subBytes(ctr);
    shiftRows(ctr);
    mixColumns(ctr);
    addRoundKey(ctr, expandedkey, round);
  }

  //3. last round, no mixColumns for this one
  subBytes(ctr);
  shiftRows(ctr);
  addRoundKey(ctr, expandedkey, round);

  //XOR with ciphertext and copy back to data
  int dataptr = tid*16;
  for(int i = 0; i < 16; i++)
  {
    data[dataptr + i] = ctr[i] ^ data[dataptr + i];
  }
}

