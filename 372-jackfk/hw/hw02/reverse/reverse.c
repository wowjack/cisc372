#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char* make_reverse(char* word){
	size_t len = strlen(word);
	char* out = (char*)malloc((len+1)*sizeof(char)); //make output char array
	out += len*sizeof(char); //set out to last element of string
	*out = '\0'; //make last element null character
	out-=sizeof(char); //reduce the pointer by 1 to start writing
	for(; *word!='\0'; word+=sizeof(char), out-=sizeof(char)){
		*out = *word; //copy all the characters in reverse order
	}
	out+=sizeof(char); //increase the pointer by one back to the original address
	return out;
}

int main(int argc, char* argv[]){
	char* out[argc];
	for(int i=1; i<argc; i++){
		out[i-1] = make_reverse(argv[i]);
	}
	for(int i=0; i<argc-1; i++){
		printf("%s\n", out[i]);
		free(out[i]);
	}
	return 0;
}
