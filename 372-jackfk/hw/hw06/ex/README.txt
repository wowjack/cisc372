1. Incorrect. y is uninitialized when each thread in the parallel region tries to print it out.
    Either make y shared or user firstprivate(y)
2. Incorrect. x not specified as private or shared and default(none) is used. Declare x to be
    firstprivate or shared, or private and initialize it within the parallal region. Also y
    has the same problem as from problem 1, it is uninitialized.
3. Incorrect. Data race on shared variable y. Make y firstprivate or private and initialize it.
4. Correct:
    Thread 0: x=10
    Thread 1: x=11
    Thread 1: y=17
    Thread 3: x=13
    Thread 2: x=12
    x=10, y=17
    [note x is back to its original value at end. y has new value]

5. Correct: 0 1 2 3 4 5 6 7 8 9 
6. Correct: 0 1 2 3 4 5 6 7 8 9
7. Incorrect. Data race when reading and writing to array a. Each iteration reads from a[i+1] 
    and writes to a[i], this can cause a data race when done in parallel.
8. Correct: 1 1 1 1 1 1 1 1 1 
9. Incorrect. Data race when reading and writing to array b. Each iteration reads from b[i+1]
    and writes to b[i], this can cause a data race when done in parallel.
10. Correct: 1 4 15 40 85 156 259 400 585 
