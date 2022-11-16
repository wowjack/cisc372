#include <stdio.h>
#include <stdlib.h>

double** create(int n, int m);
void destroy(double** a, int ySize);
void print(double** a, int ySize, int xSize);
void init(double** array, int ySize, int xSize);

int main(int argc, char* argv[]){
	int n=atof(argv[1]), m=atof(argv[2]);
	double** a = create(n,m);
	init(a, n, m);
	print(a,n,m);
	destroy(a, n);
	return 0;
}

double** create(int n, int m){
	double **outArr = (double**)malloc(n*sizeof(double*));
	for(int i=0; i<n; i++){
		outArr[i] = (double*)malloc(m*sizeof(double));
	}
	return outArr;
}
void destroy(double** a, int ySize){
	for(int i=0; i<ySize; i++){
		free(a[i]);
	}
}

void init(double** array, int ySize, int xSize){
	for(int y=0; y<ySize; y++){
		for(int x=0; x<xSize; x++){
			if(x==0 || y==0 || x==xSize-1 || y==ySize-1){
				array[y][x] = 100;
			}else{
				array[y][x] = 0;
			}
		}
	}
}

void print(double** a, int ySize, int xSize){
        for(int y=0; y<ySize; y++){
                for(int x=0; x<xSize; x++){
                        printf("%f\t", a[y][x]);
                }
                printf("\n");
        }
}

