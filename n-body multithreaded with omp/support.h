#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <errno.h>

#define GRAPH_COLUMNS 50
#define MAX_NUM_OF_STAR 20

typedef struct {
	struct timeval startTime;
	struct timeval endTime;
} Timer;

Timer timer[10];

void startTime(int i);
void stopTime(int i);
void elapsedTime(int i);

void startTime(int i) {
	//printf("Start Timer...");
	gettimeofday(&(timer[i].startTime), NULL);
}

void stopTime(int i) {
	//printf("Stop Timer...");
	gettimeofday(&(timer[i].endTime), NULL);
}

void elapsedTime(int i) {
	float elapseTime = (float) ((timer[i].endTime.tv_sec
			- timer[i].startTime.tv_sec)
			+ (timer[i].endTime.tv_usec - timer[i].startTime.tv_usec) / 1.0e6);
	printf("\t%15.6f\n", elapseTime);	fflush(stdout);
}

int findMax(int setSize, unsigned int * histogram) {
	int i, max = 0;
	for (i = 0; i < setSize; i++)
		if (histogram[i] > max)
			max = histogram[i];
	return (max > MAX_NUM_OF_STAR) ? max : MAX_NUM_OF_STAR;
}

void printHistogram(unsigned int arraySize, unsigned int * histogram, int from, int to) {
	int i = 0, j = 0;
	int max = findMax(to-from, histogram);
	for (i = 0; i <= to-from; i++) {
		printf("%d[%d]\t:", i+from, histogram[i]);
		for (j = 0; j < histogram[i]; j += max / MAX_NUM_OF_STAR)
			printf("*");
		printf("\n");
	}
}

