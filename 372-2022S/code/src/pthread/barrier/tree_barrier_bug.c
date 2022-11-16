/* Tree barrier.  Threads are arranged in a binary tree.  When arrive
   in barrier: wait for your children to tell you they have arrived,
   then tell your parent you have arrived.  Then wait for your parent
   to tell you to depart, then tell your children to depart.
 
   Author   : Stephen F. Siegel
   Date     : 2016-oct-24
   Modified : 2020-oct-14
 */
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "barrier.h"
#include "flag.h"

/* A node in the binary tree */
typedef struct node {
  int id; // thread ID # corresponding to this node
  struct node * parent; // parent node or NULL
  struct node * left; // left child node or NULL
  struct node * right; // right child node or NULL
  flag_t arrive; // used to notify parent I have arrived at barrier
  flag_t depart; // used by parent to notify me I may leave
} * Node;

struct barrier_state {
  /* The number of threads in the team participating in this barrier */
  int nthreads;
  /* Root node of binary tree. There is one node for each thread. */
  Node root;
};

struct barrier {
  barrier_state_t bs;
  Node node;
};

/* Makes a binary tree, approximately balanced.
     start_id : ID of root of tree to be created
     parent : parent of root of tree to be created
     num_nodes : number of nodes in tree to be created
   All nodes are initialized (including flags) */
static Node make_tree(int start_id, Node parent, int num_nodes) {
  if (num_nodes == 0)
    return NULL;

  Node root = (Node)malloc(sizeof(struct node));
  int num_right = (num_nodes - 1)/2;
  int num_left = num_nodes - 1 - num_right;

  assert(root);
  root->id = start_id;
  root->parent = parent;
  flag_init(&root->arrive, 0);
  flag_init(&root->depart, 0);
  root->left = make_tree(start_id+1, root, num_left);
  root->right = make_tree(start_id+num_left+1, root, num_right);
  return root;  
}

static Node find_node(Node root, int id) {
  if (root->id == id)
    return root;
  const Node right = root->right;
  if (right != NULL && right->id <= id)
    return find_node(right, id);
  const Node left = root->left;
  assert(left != NULL);
  return find_node(left, id);  
}

/* Destroys the tree rooted at root.  Semaphores are closed.  */
static void destroy_tree(Node root) {
  if (root == NULL)
    return;
  destroy_tree(root->left);
  destroy_tree(root->right);
  flag_destroy(&root->arrive);
  flag_destroy(&root->depart);
  free(root);
}

barrier_state_t barrier_state_create(int nthreads) {
  barrier_state_t bs = malloc(sizeof(struct barrier_state));
  
  assert(bs);
  bs->nthreads = nthreads;
  bs->root = make_tree(0, NULL, nthreads);
  return bs;
}

void barrier_state_destroy(barrier_state_t bs) {
  destroy_tree(bs->root);
  free(bs);
}

barrier_t barrier_create(barrier_state_t bs, int tid) {
  assert (0<=tid && tid<bs->nthreads);
  barrier_t bar = malloc(sizeof(struct barrier));
  assert(bar);
  bar->bs = bs;
  bar->node = find_node(bs->root, tid);
  return bar;
}

void barrier_destroy(barrier_t bar) {
  free(bar);
}

void barrier_wait(barrier_t bar) {
  const Node me = bar->node;
  
  // wait for notification from children:
  if (me->left != NULL)
    flag_lower(&me->left->arrive);
  if (me->right != NULL)
    flag_lower(&me->right->arrive);
  if (me->parent != NULL) {
    // notify parent my children & I have arrived:
    flag_raise(&me->arrive);
    // wait for my parent to tell me to depart:
    flag_raise(&me->depart); // BUG!
  }
  // tell my children to depart:
  if (me->left != NULL)
    flag_raise(&me->left->depart);
  if (me->right != NULL)
    flag_raise(&me->right->depart);
}
