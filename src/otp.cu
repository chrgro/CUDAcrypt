
__global__
void otp(int *v, int *k) {

		v[threadIdx.x] = v[threadIdx.x] ^ (*k);
}
