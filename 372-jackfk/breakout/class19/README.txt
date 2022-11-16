I tried running the normal parallel program on with args 1000 1000 1000 and it took about 7 seconds to run.

Just adding "#pragma omp parallel for" around each of the loops and setting the used variables to shared
helped improve runtime substantially. For the same args it reduced the runtime to around 1.2 seconds.

I added "schedule(dynamic, 10)" to the directive and it improved the runtime a bit. For the same args it
reduced the runtime to about 1.07 seconds.

I changed it to "schedule(guided, 10)" and the runtime got a bit worse. The same args rant for about 1.15 seconds.

I changed it to "schedule(guided, 1)" and nothing changed.

I tried a static scheduling but was unable to make it any better than for dynamic or guided scheduling.

Ultimately it doesn't seem to impact runtime much at all if I use dynamic or guided scheduling.
Sometimes guided would faster than dynamic, and sometimes dynamic faster than guided.
Static runtime definetly didn't work as well as dynamic or guided.
The best average runtime I could get was around 1.1 seconds, generally the runtime would vary by about .1 seconds.