#include <stdio.h>
#include <stdlib.h>

int main() {

    system ("cd");
    FILE *fp = fopen("../data/train_convert.txt", "r");
    if (fp == NULL) {
        perror("Error opening file");
        return -1;
    }else{
        printf("YAYAYAY");
    }

    // File reading code...

    fclose(fp);
    return 0;
}
