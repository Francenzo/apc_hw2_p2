#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include <vector>
#include "common.h"

#define NUM_THREADS 256
#define TILE_SIZE 32
#define BIN_TILE_SIZE 64

extern double size;
//
//  benchmarking program
//

// __global__ void clear_bins_gpu(bin_t * bin_arr, int bin_count)
// {
//   // Get thread (bin) ID
//   int tid = threadIdx.x + blockIdx.x * blockDim.x;
//   if(tid >= bin_count) return;

//   bin_arr[tid].size = 0;
// }

__global__ void malloc_particles(particle_arr_t * particles, int n)
{
  cudaMalloc((void **) particles->binNum, n * sizeof(int));
  cudaMalloc((void **) particles->x, n * sizeof(double));
  cudaMalloc((void **) particles->y, n * sizeof(double));
  cudaMalloc((void **) particles->vx, n * sizeof(double));
  cudaMalloc((void **) particles->vy, n * sizeof(double));
  cudaMalloc((void **) particles->ax, n * sizeof(double));
  cudaMalloc((void **) particles->ay, n * sizeof(double));
}

__device__ void bin_num_gpu(particle_arr_t *particle, int p_index, int size, int bin_row_size)
{
  double frac_x = (particle->x)[p_index]/size;
  double frac_y = (particle->y)[p_index]/size;
  int bin_x = frac_x * bin_row_size;
  int bin_y = frac_y * bin_row_size;
  int binNum = bin_x + ( bin_y * bin_row_size );
  (particle->binNum)[p_index] = binNum;
}

__global__ void compute_bins_gpu(particle_arr_t *particles, int n, int size, int bin_row_size)
{
  // Get thread (particle) ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= n) return;
  bin_num_gpu(particles, tid, size, bin_row_size);
}

__global__ void set_bin_gpu(particle_arr_t *particles, bin_t * bin_arr, int n, int size, int bin_row_size)
{
  // Get thread (particle) ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  // int thread_id = threadIdx.x;
  // int block_id = blockIdx.x;
  if(tid >= bin_row_size)
    return;

  // _shared_ bin_t tmp_bin_arr[64];
  int start = tid * bin_row_size;
  int end = (tid + 1) * bin_row_size;

  for (int iCount = 0; iCount < n; iCount++)
  {
    int binNum = (particles->binNum)[iCount];
    if (binNum >= start && binNum < bin_row_size * bin_row_size && binNum < end)
    {
      bin_arr[binNum].arr[bin_arr[binNum].size] = iCount;
      atomicAdd(&(bin_arr[binNum].size), 1);
    }
  }

}

__device__ void apply_force_gpu(particle_arr_t *particles, int p_index, int n_index)
{
  double dx = (particles->x)[n_index] - (particles->x)[p_index];
  double dy = (particles->y)[n_index] - (particles->y)[p_index];
  double r2 = dx * dx + dy * dy;
  if( r2 > cutoff*cutoff )
      return;
  //r2 = fmax( r2, min_r*min_r );
  r2 = (r2 > min_r*min_r) ? r2 : min_r*min_r;
  double r = sqrt( r2 );

  //
  //  very simple short-range repulsive force
  //
  double coef = ( 1 - cutoff / r ) / r2 / mass;
  (particles->ax)[p_index] += coef * dx;
  (particles->ay)[p_index] += coef * dy;

}

__global__ void compute_forces_bin_gpu(particle_arr_t *particles, bin_t * bin_arr, int n, int bin_row_size)
{
  // Get thread (particle) ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= n) return;

  (particles->ax)[tid] = (particles->ay)[tid] = 0;
  int binNum = (particles->binNum)[tid];

  // All surrounding bins in a 3x3 square
  int binsToCheck[] = {   binNum - bin_row_size - 1,
                          binNum - bin_row_size,
                          binNum - bin_row_size + 1,
                          binNum - 1,
                          binNum,
                          binNum + 1,
                          binNum + bin_row_size - 1,
                          binNum + bin_row_size,
                          binNum + bin_row_size + 1
                      };


  for(int j = 0 ; j < 9; j++)
  {
    int tmpNum = binsToCheck[j];
    if (tmpNum >= 0 && tmpNum < bin_row_size * bin_row_size)
    {
      for (int iCount = 0; iCount < bin_arr[tmpNum].size; iCount++)
        apply_force_gpu(particles, tid, bin_arr[tmpNum].arr[iCount]);
    }
  }

  // for(int j = 0 ; j < bin_row_size * bin_row_size; j++)
  // {
  
  //     for (int iCount = 0; iCount < bin_arr[j].size; iCount++)
  //       apply_force_gpu(particles[tid], particles[bin_arr[j].arr[iCount]]);
    
  // }

  // for (int iCount = 0; iCount < n; iCount++)
  //   apply_force_gpu(particles[tid], particles[iCount]);
}

// __global__ void compute_forces_gpu(particle_t * particles, int n)
// {
//   // Get thread (particle) ID
//   int tid = threadIdx.x + blockIdx.x * blockDim.x;
//   if(tid >= n) return;

//   particles[tid].ax = particles[tid].ay = 0;
//   for(int j = 0 ; j < n ; j++)
//     apply_force_gpu(particles[tid], particles[j]);

// }

__global__ void move_gpu (particle_arr_t *particles, int n, double size)
{

  // Get thread (particle) ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= n) return;

  // particle_t * p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    (particles->vx)[tid] += (particles->ax)[tid] * dt;
    (particles->vy)[tid] += (particles->ay)[tid] * dt;
    (particles->x)[tid]  += (particles->vx)[tid] * dt;
    (particles->y)[tid]  += (particles->vy)[tid] * dt;

    //
    //  bounce from walls
    //
    while( (particles->x)[tid] < 0 || (particles->x)[tid] > size )
    {
        (particles->x)[tid]  = (particles->x)[tid] < 0 ? -((particles->x)[tid]) : 2*size-(particles->x)[tid];
        (particles->vx)[tid] = -((particles->vx)[tid]);
    }
    while( (particles->y)[tid] < 0 || (particles->y)[tid] > size )
    {
        (particles->y)[tid]  = (particles->y)[tid] < 0 ? -((particles->y)[tid]) : 2*size-particles->y[tid];
        (particles->vy)[tid] = -((particles->vy)[tid]);
    }

}



int main( int argc, char **argv )
{    
  // This takes a few seconds to initialize the runtime
  cudaThreadSynchronize(); 

  if( find_option( argc, argv, "-h" ) >= 0 )
  {
      printf( "Options:\n" );
      printf( "-h to see this help\n" );
      printf( "-n <int> to set the number of particles\n" );
      printf( "-o <filename> to specify the output file name\n" );
      return 0;
  }
  
  int n = read_int( argc, argv, "-n", 1000 );
  int bin_row_size;

  int struct_size = sizeof(particle_arr_t) 
                    + n * sizeof(int) 
                    + sizeof(double) * 6 * n;

  char *savename = read_string( argc, argv, "-o", NULL );
  
  FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
  // particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
  particle_arr_t * particles;

  particles = (particle_arr_t *) malloc( sizeof(particle_arr_t));
  particles->binNum = (int *) malloc(n * sizeof(int));
  particles->x = (double *) malloc(n * sizeof(double));
  particles->y = (double *) malloc(n * sizeof(double));
  particles->vx = (double *) malloc(n * sizeof(double));
  particles->vy = (double *) malloc(n * sizeof(double));
  particles->ax = (double *) malloc(n * sizeof(double));
  particles->ay = (double *) malloc(n * sizeof(double));


  // GPU particle data structure
  // particle_t * d_particles;
  // cudaMalloc((void **) &d_particles, n * sizeof(particle_t));
  
  particle_arr_t * d_particles;
  cudaMalloc((void **) d_particles, sizeof(particle_arr_t));
  malloc_particles <<< 1,1 >>> (d_particles, n);

  


  set_size( n );
  bin_row_size = get_bin_row_size();
  int num_bins = bin_row_size * bin_row_size;
  // Particle bins
  bin_t * bin_arr = (bin_t*) malloc( num_bins * sizeof(bin_t) );
  bin_t * d_bins;
  cudaMalloc((void **) &d_bins, num_bins * sizeof(bin_t));
  // std::vector< std::vector<particle_t *> > vec_bins(num_bins);

  // init_particles( n, particles );
  init_particles_array( n, particles );

  cudaThreadSynchronize();
  double copy_time = read_timer( );

  // Copy the particles to the GPU
  // cudaMemcpy(d_particles->x, particles->x, struct_size, cudaMemcpyHostToDevice);
  // cudaMemcpy(d_particles->y, particles->y, struct_size, cudaMemcpyHostToDevice);
  // cudaMemcpy(d_particles->vx, particles->vx, struct_size, cudaMemcpyHostToDevice);
  // cudaMemcpy(d_particles->vy, particles->vy, struct_size, cudaMemcpyHostToDevice);
  // cudaMemcpy(d_particles->ax, particles->ax, struct_size, cudaMemcpyHostToDevice);
  // cudaMemcpy(d_particles->ay, particles->ay, struct_size, cudaMemcpyHostToDevice);

  cudaMemcpy(d_particles, particles, struct_size, cudaMemcpyHostToDevice);


#define GPU_BINS
#ifdef GPU_BINS
  memset(bin_arr, 0x0, num_bins*sizeof(bin_t) );
  cudaMemset(d_bins, 0, num_bins*sizeof(bin_t) );
  cudaMemcpy(d_bins, bin_arr, num_bins * sizeof(bin_t), cudaMemcpyHostToDevice);
#endif

  cudaThreadSynchronize();
  copy_time = read_timer( ) - copy_time;
  
  //
  //  simulate a number of time steps
  //
  cudaThreadSynchronize();
  double simulation_time = read_timer( );

#ifdef GPU_BINS
  printf("Using bins...\r\n");
#endif

  int grid = (n + NUM_THREADS - 1) / NUM_THREADS;

  for( int step = 0; step < NSTEPS; step++ )
  {
    int grid_bins = (bin_row_size + NUM_THREADS - 1) / NUM_THREADS;

#ifdef GPU_BINS
    //
    // Put particles into bins
    //
    compute_bins_gpu <<< grid, NUM_THREADS >>> (d_particles, n, size, bin_row_size);

    //
    // Put particles into bins
    //
    set_bin_gpu <<< grid, NUM_THREADS >>> (d_particles, d_bins, n, size, bin_row_size);

    //
    //  compute bins
    //
    compute_forces_bin_gpu <<< grid, NUM_THREADS >>> (d_particles, d_bins, n, bin_row_size);
#else

    //
    //  compute forces
    //
	  compute_forces_gpu <<< grid, NUM_THREADS >>> (d_particles, n);


    // cudaMemcpy(particles, d_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost);
    // printf("particle[%i] = %f\r\n", 1, particles[1].y);
#endif
        
    //
    //  move particles
    //
	  move_gpu <<< grid, NUM_THREADS >>> (d_particles, n, size);

#ifdef GPU_BINS
    // Check if all are there
    // cudaMemcpy(bin_arr, d_bins, n * sizeof(bin_t), cudaMemcpyDeviceToHost);
    // for (int jCount =0; jCount < n; jCount++)
    // {
    //   bool isFound = false;
    //   for (int iCount = 0; iCount < bin_row_size * bin_row_size; iCount++)
    //   {
    //     for (int lCount = 0; lCount < bin_arr[iCount].size; lCount++)
    //     {
    //       if (bin_arr[iCount].arr[lCount] == jCount)
    //       {
    //         isFound = true;
    //         break;
    //       }
    //     }
    //     if (isFound)
    //     {
    //       break;
    //     }
    //   }
    //   if (!isFound)
    //   {
    //     printf("particle %i not found in bin\r\n", jCount);
    //   }
    // }

    // int binTotes = 0;
    // for (int iCount = 0; iCount < bin_row_size * bin_row_size; iCount++)
    // {
    //   binTotes += bin_arr[iCount].size;
    // }
    // printf("Totes size = %i, bin[0].size = %i\r\n", binTotes, bin_arr[0].size);

    // exit(0);
    
    // Set bin array to null
    cudaMemset(d_bins, 0, num_bins*sizeof(bin_t));

#endif

    // cudaMemcpy(particles, d_particles, struct_size, cudaMemcpyDeviceToHost);
    // printf("particle[%i].x = %f\r\n", 100, particles->x[100]);
        
    //
    //  save if necessary
    //
    if( fsave && (step%SAVEFREQ) == 0 ) {
      // Copy the particles back to the CPU
      cudaMemcpy(particles, d_particles, struct_size, cudaMemcpyDeviceToHost);

      save_array( fsave, n, particles);
    }
  }

  cudaThreadSynchronize();
  simulation_time = read_timer( ) - simulation_time;
  
  printf( "CPU-GPU copy time = %g seconds\n", copy_time);
  printf( "n = %d, simulation time = %g seconds\n", n, simulation_time );
  
  free( particles->x );
  free( particles->y );
  free( particles->vx );
  free( particles->vy );
  free( particles->ax );
  free( particles->ay );
  cudaFree(d_particles);
  if( fsave )
      fclose( fsave );
  
  return 0;
}
