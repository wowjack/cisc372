/* Filename : integral_mpi.c
   Author   : Stephen F. Siegel, University of Delaware
   Date     : 16-apr-2020

   Parallel MPI version of integral.c.  Based on manager-worker
   pattern.  Manager generates a set of tasks organized in a tree.
   Tasks are distributed to workers.  New tasks are assigned as soon
   as a worker returns a result.

   The set of tasks is static: it is computed once, at the beginning,
   by the manager.  The number of tasks is at most subscription_ratio
   (100) times the number of workers.
*/
#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

// Type definitions...

typedef long double T; // floating point type we will use

struct task_s { // an integration task
  T a; // left end-point
  T b; // right end-point
  T fa; // f(a)
  T fb; // f(b);
  T area; // area of trapezoid with base [a,b]
  T tol; // tolerance needed for this interval
};

typedef struct node_s * node_t;

struct node_s {  // a node in the rooted tree of tasks
  struct task_s task; // the task associated to this node
  node_t parent; // node that spawned this one
  node_t child0; // node responsible for [a,c]
  node_t child1; // node responsible for [c,b]
  T result; // integral(child0) + integral(child1)
  bool computed; // has result been computed?
};

// Global variables...

const int subscription_ratio = 100; // num tasks per worker
const MPI_Comm comm = MPI_COMM_WORLD;
const MPI_Datatype float_dt = MPI_LONG_DOUBLE; // MPI datatype for T
MPI_Datatype task_dt; // MPI datatype corresponding to struct task_s
int nprocs, rank; // number of procs, rank of this proc
int count = 0; // number of integration steps done on this proc
node_t * nodes = NULL; // the nodes (used on manager only)
int numNodes; // number of nodes in manager's task tree (manager only)
int first; // index of first node that needs to be sent out (manager only)

// Function definitions...

static inline T f(const T x) { return 4/(1+x*x); }

/* Perform the computations of one step in the adaptive quadrature
   process.
   
   Given old values a, b, fa=f(a), fb=f(b), tol (desired tolerance),
   area (of trapezoid specified by a, b, fa, fb), this function
   computes: c=(a+b)/2, fc=f(c), leftArea (area of left trapezoid
   specified by a, c, fa, fc), rightArea (area of right trapezoid
   specified by c, b, fc, fb), area2=leftArea+rightArea, and
   tolDiv2=tol/2.0 (only computed if not converged).

   area2 is the newly computed, more precise approximation to the
   integral on [a,b].  If |area2-area|<=tol, the interval is
   considered converged and true is returned.  Otherwise, false is
   returned.
*/
static bool inline
converged(const T a, const T b, const T fa, const T fb, const T tol,
	  const T area, T * const c, T * const fc, T * const leftArea,
	  T * const rightArea, T * const area2, T * tolDiv2) {
  const T delta = b - a;

  if (tol == 0) {
    printf("Insufficent precision to obtain desired tolerance.\n");
    exit(1);
  }
  count++;
  *c = a + delta/2;
  *fc = f(*c);
  *leftArea = (fa+*fc)*delta/4;
  *rightArea = (*fc+fb)*delta/4;
  *area2 = *leftArea + *rightArea;
  if (fabsl(*area2 - area) > tol) {
    *tolDiv2 = tol/2;
    return false;
  }
  return true;
}

/* Recursive function to compute the approximate integral on [a,b] to
   within a specified tolerance.  Given endpoints a, b, and fa=f(a),
   fb=f(b), area (of trapezoid specified by a, b, fa, fb), and desired
   tolerance tol, the function repeatedly subdivides intervals until
   the desired tolerance has been met, and returns the estimate of the
   integral. */
static T integrate(const T a, const T b, const T fa, const T fb,
		   const T area, const T tol) {
  T c, fc, leftArea, rightArea, area2, tolDiv2;

  return converged(a, b, fa, fb, tol, area, &c, &fc, &leftArea, &rightArea,
		   &area2, &tolDiv2) ? area2 :
    integrate(a, c, fa, fc, leftArea, tolDiv2) +
    integrate(c, b, fc, fb, rightArea, tolDiv2);
}

/* Creates a new node in the task tree on the manager process.  The
   node fields are initialized with the given values.  The two
   children fields will be NULL, and computed will be false. */
static inline node_t
new_node(const node_t parent, const T a, const T b,
	 const T fa, const T fb, const T area, const T tol) {
  node_t node = malloc(sizeof(struct node_s));

  assert(node);
  node->task.a = a;
  node->task.b = b;
  node->task.fa = fa;
  node->task.fb = fb;
  node->task.area = area;
  node->task.tol = tol;
  node->parent = parent;
  node->child0 = node->child1 = NULL;
  node->computed = false;
  return node;
}

/* This function is called once a result has been returned for a node.
   The result field is set to the given value.  Furthermore, the
   function walks up the tree setting the results of ancestor nodes,
   for any ancestor that now has both children nodes computed.  */
static inline void set_result(node_t node, T value) {
  node->result = value;
  node->computed = true;
  for (node = node->parent;
       node != NULL && node->child0->computed && node->child1->computed;
       node = node->parent) {
    node->result = node->child0->result + node->child1->result;
    node->computed = true;
  }
}

/* Creates the task tree.  Called once, by the manager, at the
   beginning.  The tasks are the uncomputed leaf nodes of the tree:
   these are the tasks that will be distributed to the workers.
   This function initializes global vars nodes, first.*/
static void create_nodes(const T a0, const T b0, const T tol0) {
  const int goal = (nprocs - 1) * subscription_ratio; // want this many tasks
  const T fa0 = f(a0), fb0 = f(b0);
  // The actual length of the nodes array, though not all cells are used:
  int numNodesMax = goal > 2 ? goal : 2;
  
  nodes = malloc(numNodesMax * sizeof(node_t));
  assert(nodes);
  nodes[0] = new_node(NULL, a0, b0, fa0, fb0, (fa0+fb0*(b0-a0))/2, tol0);
  numNodes = 1;
  /* Loop invariant:
       for all i:0..first-1: nodes[i].computed || !leaf_node(nodes[i])
       for all i:first..numNodes-1: !nodes[i].computed && leaf_node(nodes[i])
    Where leaf_node(u) <==> u.child0==NULL && u.child1==NULL.
    Note: first is a global variable.  It is initialized here.  */
  for (first=0; numNodes - first < goal && first < numNodes; first++) {
    const node_t node = nodes[first]; // the first unexplored node
    const T a   = node->task.a,        b = node->task.b,
            fa  = node->task.fa,      fb = node->task.fb,
            tol = node->task.tol,   area = node->task.area;
    T c, fc, leftArea, rightArea, area2, tolDiv2;

    if (converged(a, b, fa, fb, tol, area, &c, &fc, &leftArea, &rightArea,
		  &area2, &tolDiv2)) {
      set_result(node, area2);
    } else {
      node->child0 = new_node(node, a, c, fa, fc, leftArea, tolDiv2);
      node->child1 = new_node(node, c, b, fc, fb, rightArea, tolDiv2);
      if (numNodes+2 > numNodesMax) {
	numNodesMax *= 2;
	nodes = realloc(nodes, numNodesMax * sizeof(node_t));
	assert(nodes);
      }
      assert(numNodes+2 <= numNodesMax);
      nodes[numNodes++] = node->child0;
      nodes[numNodes++] = node->child1;
    }
  }
#ifdef DEBUG
  printf("Manager: numNodesMax=%d, numNodes=%d, first=%d\n",
	 numNodesMax, numNodes, first);
  printf("Manager integrate calls: %d\n", count);
  fflush(stdout);
#endif
}

static void destroy_nodes() {
  for (int i=0; i<numNodes; i++) free(nodes[i]);
  free(nodes);
}

/* Each worker process executes this function.  Receives an
   integration tasks from manager, completes the tasks and sends
   result back to manager, and repeats, until a termination signal is
   received. */
static void worker() {
  MPI_Status status;
  struct task_s task;
  int taskCount = 0;
  T result;

  while (1) {
    MPI_Recv(&task, 1, task_dt, 0, MPI_ANY_TAG, comm, &status);
    if (status.MPI_TAG == 0) break; // the termination signal
    taskCount++;
    result = integrate(task.a, task.b, task.fa, task.fb, task.area, task.tol);
    MPI_Send(&result, 1, float_dt, 0, status.MPI_TAG, comm);
  }
#ifdef DEBUG
  printf("Worker %d completed %d tasks and %d calls to integrate.\n",
	 rank, taskCount, count);
  fflush(stdout);
#endif
  MPI_Reduce(&count, NULL, 1, MPI_INT, MPI_SUM, 0, comm);
}

/* The manager process runs this function.  Hands out the tasks to
   workers, waits for them to come back, processes the results, and
   repeats until done. */
static void manager(const T a, const T b, const T tol) {
  int wid; // worker ID: the rank of a worker process
  int tid; // task ID: an index into nodes array
  int total_count; // sum over all procs of count (num. integration steps)
  MPI_Status status;
  T result;

  create_nodes(a, b, tol);
  // send one task to each worker. tasks are indexed first .. numNodes-1
  // note that first is a global variable, initialized in create_nodes
  for (wid=1, tid=first; wid<nprocs && tid<numNodes; wid++, tid++)
    MPI_Send(&nodes[tid]->task, 1, task_dt, wid, tid, comm);
  // wait for a response, send new task, repeat until all tasks distributed...
  for (; tid < numNodes; tid++) {
    MPI_Recv(&result, 1, float_dt, MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &status);
    MPI_Send(&nodes[tid]->task, 1, task_dt, status.MPI_SOURCE, tid, comm);
    set_result(nodes[status.MPI_TAG], result);
  }
  // get last result from each worker; send terminate signals to all...
  for (wid=1; wid<nprocs; wid++) {
    if (wid <= numNodes - first) {
      MPI_Recv(&result, 1, float_dt, wid, MPI_ANY_TAG, comm, &status);
      set_result(nodes[status.MPI_TAG], result);
    }
    MPI_Send(NULL, 0, task_dt, wid, 0, comm); // send termination signal
  }
  MPI_Reduce(&count, &total_count, 1, MPI_INT, MPI_SUM, 0, comm);
  assert(nodes[0]->computed);
  printf("Number of intervals: %d\n", total_count);
  printf("Result: %4.20Lf\n", nodes[0]->result);
  destroy_nodes();
}

int main() {
  MPI_Init(NULL, NULL);
  MPI_Comm_size(comm, &nprocs);
  assert(nprocs >= 2); // need at least a manager and a worker
  MPI_Comm_rank(comm, &rank);
  MPI_Type_contiguous(6, float_dt, &task_dt);
  MPI_Type_commit(&task_dt);
  if (rank == 0) manager(0, 1, 1e-18); else worker();
  MPI_Type_free(&task_dt);
  MPI_Finalize();
}
