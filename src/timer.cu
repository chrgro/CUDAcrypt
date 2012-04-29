#include "timer.h"
#include "stdio.h"

bool timer_running = false;
cudaEvent_t timer_start, timer_stop; 
float timer_time; 

void timerStart() {
	if (!timer_running) {
		timer_running = true;
		
		cudaEventCreate(&timer_start); 
		cudaEventCreate(&timer_stop); 
		cudaEventRecord( timer_start, 0 ); 
		
	} else {
		printf("Invalid timer start/stop order\n");
		exit(-1);
	}

}

float timerStop() {
	if (timer_running) {
		timer_running = false;
		
		cudaEventRecord( timer_stop, 0 ); 
		cudaEventSynchronize( timer_stop ); 
		cudaEventElapsedTime( &timer_time, timer_start, timer_stop ); 
		cudaEventDestroy( timer_start ); 
		cudaEventDestroy( timer_stop );
		
		return timer_time;
	} else {
		printf("No timer running \n");
		exit(-1);
	}
}