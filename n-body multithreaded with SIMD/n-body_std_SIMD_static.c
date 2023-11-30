#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "support.h"
#include <omp.h>
#include <string.h>
#include <immintrin.h>

#define VECTOR_SIZE 8 // Number of elements per SIMD vector

// Chnage this value to reflect your ID number
#define ID 1031373
#define END -1
typedef struct
{
    double x, y, z;
} vector;

int bodies, timeSteps;
int SimulationTime = 0;
double *masses, GravConstant;
double *positions_x, *positions_y, *positions_z;
double *velocities_x, *velocities_y, *velocities_z;
vector *accelerations;

double getRND()
{
    return (100.0 - rand() % 200) / 50.0;
}
void initiateSystemRND(int bodies)
{
    int i;
    srand(ID);

    // Align memory to a 64-byte boundary
    masses = (double *)_mm_malloc(bodies * sizeof(double), 64);
    positions_x = (double *)_mm_malloc(bodies * sizeof(double), 64);
    positions_y = (double *)_mm_malloc(bodies * sizeof(double), 64);
    positions_z = (double *)_mm_malloc(bodies * sizeof(double), 64);
    velocities_x = (double *)_mm_malloc(bodies * sizeof(double), 64);
    velocities_y = (double *)_mm_malloc(bodies * sizeof(double), 64);
    velocities_z = (double *)_mm_malloc(bodies * sizeof(double), 64);
    accelerations = (vector *)_mm_malloc(bodies * sizeof(vector), 64);

    GravConstant = 0.01;
    for (i = 0; i < bodies; i++)
    {
        masses[i] = 0.4; //(rand()%100)/100.0;//0.4 Good Example
        positions_x[i] = getRND();
        positions_y[i] = getRND();
        positions_z[i] = getRND();
        velocities_x[i] = getRND() / 5.0; // 0.0;
        velocities_y[i] = getRND() / 5.0; // 0.0;
        velocities_z[i] = getRND() / 5.0; // 0.0;
    }
}

void resolveCollisions(int noOfThreads)
{
    int i, j, changes;
    double dx, dy, dz, md;

    // Matrix to hold the changes to be made in the form of indexes - velocityPositions[bodies-1][bodies]
    int **velocityPositions = (int **)malloc((bodies - 1) * sizeof(int *));
    for (i = 0; i < (bodies - 1); i++)
    {
        velocityPositions[i] = (int *)malloc(bodies * sizeof(int));
    }
    // Set the private variables to be used, so that threads don't mix up. Also seth number of threads to be used.
    #pragma omp parallel private(i, j, dx, dy, dz, md) num_threads(noOfThreads) shared(bodies, masses, noOfThreads, changes, positions_x, positions_y, positions_z)
    {
    // Set the static parallelization of the for loop. Add reduction clause to changes so that the
    // changes made to that variable is visible to each thread
    #pragma omp for schedule(static, VECTOR_SIZE) reduction(+ : changes)
        for (i = 0; i < bodies - 1; i++)
        {
            changes = 0;
            for (j = i + 1; j < bodies; j++)
            {
                md = masses[i] + masses[j];
                dx = fabs(positions_x[i] - positions_x[j]);
                dy = fabs(positions_y[i] - positions_y[j]);
                dz = fabs(positions_z[i] - positions_z[j]);
                // if(positions_x[i]==positions_x[j] && positions_y[i]==positions_y[j] && positions_z[i]==positions_z[j]){
                if (dx < md && dy < md && dz < md)
                {
                    velocityPositions[i][changes] = j; // Save index for later swapping
                    changes++;                         // Increment number of changes for this body
                }
            }
            velocityPositions[i][changes] = END; // Set the end position of changes for this body
        }
    }
    #pragma omp barrier // Wait until all threads are done
    #pragma omp single  // Get a thread to do the swaps
    {
        for (i = 0; i < bodies - 1; i++)
        {
            for (j = 0; velocityPositions[i][j] != -1; j++)
            {
                double vel_x = velocities_x[i];
                double vel_y = velocities_y[i];
                double vel_z = velocities_z[i];
                velocities_x[i] = velocities_x[velocityPositions[i][j]];
                velocities_y[i] = velocities_y[velocityPositions[i][j]];
                velocities_z[i] = velocities_z[velocityPositions[i][j]];
                velocities_x[velocityPositions[i][j]] = vel_x;
                velocities_y[velocityPositions[i][j]] = vel_y;
                velocities_z[velocityPositions[i][j]] = vel_z;
            }
        }
    }
    for (i = 0; i < (bodies - 1); i++)
    {
        free(velocityPositions[i]);
    }
    free(velocityPositions);
}

void computeAccelerations(int noOfThreads)
{
    int i, j;
    #pragma omp parallel for schedule(static, bodies / noOfThreads) num_threads(noOfThreads) shared(accelerations, positions_x, positions_y, positions_z, masses) private(i, j)
    for (i = 0; i < bodies; i++)
    {
        // Initialize acceleration vectors for the current batch of bodies
        __m512d ax = _mm512_setzero_pd();
        __m512d ay = _mm512_setzero_pd();
        __m512d az = _mm512_setzero_pd();

        // Load position vector for 8 doubles from current (all 8 will be the same)
        __m512d pix = _mm512_set1_pd(positions_x[i]);
        __m512d piy = _mm512_set1_pd(positions_y[i]);
        __m512d piz = _mm512_set1_pd(positions_z[i]);

        // Compute acceleration for each body in the batch
        for (j = 0; j < bodies; j += VECTOR_SIZE)
        {
            // Load positions and masses for the current batch of bodies
            __m512d pjx = _mm512_load_pd(&positions_x[j]); // load 8 consecutive doubles starting from the address of positions_x[j]
            __m512d pjy = _mm512_load_pd(&positions_y[j]); // load 8 consecutive doubles starting from the address of positions_y[j]
            __m512d pjz = _mm512_load_pd(&positions_z[j]); // load 8 consecutive doubles starting from the address of positions_z[j]
            __m512d m = _mm512_load_pd(&masses[j]);        // load 8 consecutive doubles starting from the address of masses[j]

            __mmask8 mask;
            if (i >= j && i <= j + 7)
            {
                // Create a mask for the current batch of bodies, where the elements are 0 if j equals i, but are 1 otherwise (for if(i!=j))
                mask = (1 << VECTOR_SIZE) - 1;
                mask &= ~(1 << i % VECTOR_SIZE);
            }

            // Compute distances for sij (Subtract 8 position doubles at the same time, for x,y and z)
            __m512d sij_x = _mm512_sub_pd(pix, pjx);
            __m512d sij_y = _mm512_sub_pd(piy, pjy);
            __m512d sij_z = _mm512_sub_pd(piz, pjz);

            // Compute distances for sji (Subtract 8 position doubles at the same time, for x,y and z)
            __m512d sji_x = _mm512_sub_pd(pjx, pix);
            __m512d sji_y = _mm512_sub_pd(pjy, piy);
            __m512d sji_z = _mm512_sub_pd(pjz, piz);

            // Get mod3
            __m512d inside_sqrt = _mm512_add_pd(_mm512_add_pd(_mm512_mul_pd(sij_x, sij_x), _mm512_mul_pd(sij_y, sij_y)), _mm512_mul_pd(sij_z, sij_z));
            __m512d mod = _mm512_sqrt_pd(inside_sqrt);
            __m512d mod3 = _mm512_mul_pd(mod, _mm512_mul_pd(mod, mod));

            // Compute accelerations for the current batch of bodies
            __m512d s = _mm512_mul_pd(_mm512_set1_pd(GravConstant), _mm512_div_pd(m, mod3));

            // Calculate S
            __m512d S_x = _mm512_mul_pd(s, sji_x);
            __m512d S_y = _mm512_mul_pd(s, sji_y);
            __m512d S_z = _mm512_mul_pd(s, sji_z);

            if (i >= j && i <= j + 7)
            {
                // Reduce S using the mask to only store the result for the j's that are not equal to i
                S_x = _mm512_maskz_mov_pd(mask, S_x);
                S_y = _mm512_maskz_mov_pd(mask, S_y);
                S_z = _mm512_maskz_mov_pd(mask, S_z);
            }

            // Add to the accelerations of the ith body the results from these 8 'j' bodies
            ax = _mm512_add_pd(ax, S_x);
            ay = _mm512_add_pd(ay, S_y);
            az = _mm512_add_pd(az, S_z);
        }

        // Finally, store the accelerations for the body i
        accelerations[i].x = _mm512_reduce_add_pd(ax);
        accelerations[i].y = _mm512_reduce_add_pd(ay);
        accelerations[i].z = _mm512_reduce_add_pd(az);
    }
}

void computeVelocities()
{
    int i;
    for (i = 0; i < bodies; i++)
    {
        // velocities[i] = addVectors(velocities[i],accelerations[i]);
        vector ac = {velocities_x[i] + accelerations[i].x, velocities_y[i] + accelerations[i].y, velocities_z[i] + accelerations[i].z};
        velocities_x[i] = ac.x;
        velocities_y[i] = ac.y;
        velocities_z[i] = ac.z;
    }
}

void computePositions()
{
    int i;
    for (i = 0; i < bodies; i++)
    {
        // positions[i] = addVectors(positions[i],addVectors(velocities[i],scaleVector(0.5,accelerations[i])));
        vector sc = {0.5 * accelerations[i].x, 0.5 * accelerations[i].y, 0.5 * accelerations[i].z};
        vector ac = {velocities_x[i] + sc.x, velocities_y[i] + sc.y, velocities_z[i] + sc.z};
        vector bc = {positions_x[i] + ac.x, positions_y[i] + ac.y, positions_z[i] + ac.z};
        positions_x[i] = bc.x;
        positions_y[i] = bc.y;
        positions_z[i] = bc.z;
    }
}

void printBodiesInfo(FILE *lfp, FILE *dfp)
{
    int j;
    for (j = bodies - 10; j < bodies; j++)
        fprintf(lfp, "Body%d %f\t: %lf\t%f\t%lf\t|\t%lf\t%lf\t%lf\n", j + 1, masses[j], positions_x[j], positions_y[j], positions_z[j], velocities_x[j], velocities_y[j], velocities_z[j]);
    fprintf(lfp, "-------------------------------------------------------------------------------------------\n");
    for (j = bodies - 10; j < bodies; j++)
        fprintf(stdout, "Body%d %f\t: %lf\t%f\t%lf\t|\t%lf\t%lf\t%lf\n", j + 1, masses[j], positions_x[j], positions_y[j], positions_z[j], velocities_x[j], velocities_y[j], velocities_z[j]);
    fprintf(stdout, "-------------------------------------------------------------------------------------------\n");
}

void simulate(int noOfThreads)
{
    SimulationTime++;
    // Functions for static parallelization
    computeAccelerations(noOfThreads);
    computePositions();
    computeVelocities();
    resolveCollisions(noOfThreads);
}

int main(int argc,char* argv[]){
	int i;
	FILE* lfp = fopen("./outputs/logfile.txt","w");
	FILE* dfp = fopen("./outputs/data.dat","w");
	if (lfp == NULL || dfp == NULL){
		printf("Please create the ./outputs directory\n");
		return -1;
	}
	if (argc == 3){
		timeSteps = atoi(argv[1]);
		bodies = atoi(argv[2]);
	}else{
		printf("%%*** RUNNING WITH DEFAULT VALUES ***\n");
		timeSteps = 10000;
		bodies = 200;
	}
	initiateSystemRND(bodies);
	//initiateSystem("input.txt");
	fprintf(stdout,"Running With %d Bodies for %d timeSteps. Initial state:\n",bodies,timeSteps);
	fprintf(stderr,"Running With %d Bodies for %d timeSteps. Initial state:\n",bodies,timeSteps);
	fprintf(lfp,"Running With %d Bodies for %d timeSteps. Initial state:\n",bodies,timeSteps);
	fprintf(lfp,"Body   \t\t\t:\t\tx\t\ty\t\t\tz\t\t|\t\tvx\t\t\tvy\t\t\tvz\t\t\n");
	printBodiesInfo(lfp, dfp);
	int no_of_threads = atoi(argv[3]);
	startTime(0);	
	for(i=0;i<timeSteps;i++){
		simulate(no_of_threads);
		#ifdef DEBUG
			int j;
			//printf("\nCycle %d\n",i+1);
			for(j=0;j<bodies;j++)
				fprintf(dfp,"%d\t%d\t%lf\t%lf\t%lf\n",i,j, positions[j].x,positions[j].y,positions[j].z);
		#endif
	}
	stopTime(0);
	fprintf(lfp,"\nLast Step = %d\n",i);
	printBodiesInfo(lfp, dfp);
	printf("\nSimulation Time:");elapsedTime(0);
	elapsedTime(0);
    fclose(lfp);
    fclose(dfp);
    free(positions_x);
    free(positions_y);
    free(positions_z);
    free(velocities_x);
    free(velocities_y);
    free(velocities_z);
    free(accelerations);
    free(masses);
	return 0;
}