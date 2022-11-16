#include <stdio.h>
#include <stdlib.h>

void print_array(int xSize, int ySize, double[ySize][xSize]);

int main(int argc, char *argv[]){
	int n=atoi(argv[1]), m=atoi(argv[2]);
	double a[n][m];
	//Initialize top and bottom rows to 100s
	for(int i=0; i<m; i++){
		a[i][0]=100;
		a[i][n-1]=100;
	}
	//initialize left and right columns to 100s
	for(int i=0; i<n; i++){
		a[0][i]=100;
		a[m-1][i]=100;
	}
	for(int y=1; y<n-1; y++){
		for(int x=1; x<m-1; x++){
			a[x][y] = 0;
		}
	}
	print_array(m, n, a);
	return 0;
}

void print_array(int xSize, int ySize, double a[ySize][xSize]){
	for(int x=0; x<xSize; x++){
		for(int y=0; y<ySize; y++){
			printf("%f\t", a[x][y]);
		}
		printf("\n");
	}
}
