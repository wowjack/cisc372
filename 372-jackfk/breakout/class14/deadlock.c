#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

pthread_mutex_t mutex;
pthread_t thread;

void* func(void* arg){
	printf("Thread waiting for mutex to unlock.\n");
	pthread_mutex_lock(&mutex);
	printf("Mutex unlocked, hello!\n");
	pthread_mutex_unlock(&mutex);
}

//func can only return when mutex is unlocked
//But main thread locks mutex and only unlocks when thread func returns
//So deadlock happens
int main(int argc, char* argv[]){

	pthread_mutex_init(&mutex, NULL);

	printf("Locking mutex from main thread.\n");
	pthread_mutex_lock(&mutex);

	pthread_create(&thread, NULL, func, NULL);
	pthread_join(thread, NULL);

	pthread_mutex_unlock(&mutex);

	printf("Finished\n");


	return 0;
}
