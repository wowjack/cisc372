/* This version of bank has one deposit thread and one withdraw
   thread.  The depositor won't deposit unless the balance is small
   enough.  The withdrawer won't withdraw unless the balance is big
   enough.  */
#include <stdio.h>
#include <pthread.h>

// keep the variable bal between 0 and 10
const int max = 10;
int bal = 0;
pthread_mutex_t mutex;
pthread_cond_t balLT10, balGT0;

void * deposit_thread(void * arg) {
  while (1) {
    pthread_mutex_lock(&mutex);
    while (!(bal<max)) {
      printf("Depositor waiting...\n"); fflush(stdout);
      pthread_cond_wait(&balLT10, &mutex);
      printf("Depositor awakened.\n"); fflush(stdout);
    }
    // now I know bal<10 and I have the lock
    bal++;
    printf("Deposit made.  Balance = %d\n", bal); fflush(stdout);
    pthread_cond_signal(&balGT0);
    pthread_mutex_unlock(&mutex);    
  }
}

void * withdraw_thread(void * arg) {
  while (1) {
    pthread_mutex_lock(&mutex);
    while (!(bal>0)) {
      printf("Withdrawer waiting...\n"); fflush(stdout);
      pthread_cond_wait(&balGT0, &mutex);
      printf("Withdrawer awakened.\n"); fflush(stdout);
    }
    // now I know bal>0 and I have the lock
    bal--;
    printf("Withdraw made.  Balance = %d\n", bal); fflush(stdout);
    pthread_cond_signal(&balLT10);
    pthread_mutex_unlock(&mutex);    
  }
}

int main() {
  pthread_mutex_init(&mutex, NULL);
  pthread_cond_init(&balLT10, NULL);
  pthread_cond_init(&balGT0, NULL);
  pthread_t depositor, withdrawer;
  pthread_create(&depositor, NULL, deposit_thread, NULL);
  pthread_create(&withdrawer, NULL, withdraw_thread, NULL);
  pthread_join(depositor, NULL);
  pthread_join(withdrawer, NULL);
  pthread_cond_destroy(&balGT0);
  pthread_cond_destroy(&balLT10);
  pthread_mutex_destroy(&mutex);
}
