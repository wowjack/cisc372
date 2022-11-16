
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <assert.h>

int naccount; // number of accounts
int ndeposit; // number of depositors
int nwithdraw; // number of withdrawers
float * balances;
pthread_mutex_t * locks;
pthread_cond_t * conds;

int myrand(int max) {
  return rand() % max;
}

void * run_depositor(void * arg) {
  int tid = *((int*)arg);

  printf("Hello from depositor %d\n", tid);
  fflush(stdout);
  while (1) {
    float amount = myrand(100);
    int account = myrand(naccount);
    
    pthread_mutex_lock(locks + account);
    balances[account] += amount;
    printf("Depositor %d deposited %.2f to account %d. New balance: %.2f\n",
	   tid, amount, account, balances[account]);
    fflush(stdout);
    pthread_cond_broadcast(conds + account);
    pthread_mutex_unlock(locks + account);
  }
  return NULL;
}

void * run_withdrawer(void * arg) {
  int tid = *((int*)arg);

  printf("Hello from withdrawer %d\n", tid);
  fflush(stdout);
  while (1) {
    float amount = myrand(200);
    int account = myrand(naccount);

    pthread_mutex_lock(locks + account);
    while (balances[account] < amount) {
      printf("Withdrawer %d: WAITING to withdraw %.2f from account %d. Balance=%.2f\n",
	     tid, amount, account, balances[account]);
      fflush(stdout);
      pthread_cond_wait(conds + account, locks + account);
    }
    balances[account] -= amount;
    printf("Withdrawer %d withdrew %.2f from account %d. New balance: %.2f\n",
	   tid, amount, account, balances[account]);
    fflush(stdout);
    pthread_mutex_unlock(locks + account);
  }
  return NULL;
}

// 1: number of accounts
// 2: number of depositors
// 3: number of withdrawers
int main(int argc, char * argv[]) {
  assert (argc >= 4);
  naccount = atoi(argv[1]);
  ndeposit = atoi(argv[2]);
  nwithdraw = atoi(argv[3]);
  balances = (float *) malloc(naccount * sizeof(float));
  locks = (pthread_mutex_t *) malloc (naccount * sizeof(pthread_mutex_t));
  conds = (pthread_cond_t *) malloc(naccount * sizeof(pthread_cond_t));
  for (int i=0; i<naccount; i++) balances[i] = 100.00;

  pthread_t withdrawers[nwithdraw], depositors[ndeposit];
  int dids[ndeposit], wids[nwithdraw];

  for (int i=0; i<ndeposit; i++) dids[i] = i;
  for (int i=0; i<nwithdraw; i++) wids[i] = i;
  for (int i=0; i<naccount; i++) {
    pthread_mutex_init(locks + i, NULL);
    pthread_cond_init(conds + i, NULL);
  }
  for (int i=0; i<ndeposit; i++)
    pthread_create(depositors + i, NULL, run_depositor, dids+i);
  for (int i=0; i<nwithdraw; i++)
    pthread_create(withdrawers + i, NULL, run_withdrawer, wids+i);
  for (int i=0; i<ndeposit; i++)
    pthread_join(depositors[i], NULL);
  for (int i=0; i<nwithdraw; i++)
    pthread_join(withdrawers[i], NULL);
  free(locks);
  free(conds);
  free(balances);
}
