#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>

double s;

struct argStruct{
	int FIRST;
	int NUM_OWNED;
};
void* getMax(void* arg){
	struct argStruct* args = arg;
	int FIRST = args->FIRST, NUM_OWNED = args->NUM_OWNED;
	free(arg);
	printf("%d, %d\n", FIRST, NUM_OWNED);
	//Compute max of array and return
}


int main(int argc, char* argv[]){
	if(argc<2) {printf("Needs 1 numerical arg\n"); return 0;}

	int arrSize = 10;
	double nums[arrSize];

	int numThreads = atoi(argv[1]);
	pthread_t threads[numThreads];
	for(int i=0; i<numThreads; i++){
		struct argStruct* args = malloc(sizeof(struct argStruct));
		args->FIRST = i*arrSize/numThreads; args->NUM_OWNED = ((i+1)*arrSize/numThreads)-(i*arrSize/numThreads);
		pthread_create(&threads[i], NULL, getMax, args);
	}
	for(int i=0; i<numThreads; i++){
		pthread_join(threads[i], NULL);
	}

	for(int i=0; i<arrSize; i++){
		nums[i] = (double)i;
	}

	fflush(stdout);

	

	return 0;
}
