#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct{
	int value;
	char suit[10];
}Card;

void print_card(Card);

int main(int argc, char* argv[]){
	
	return 0;
}

void print_card(Card card){
	printf("%d of %s\n", card.value, card.suit);
}
