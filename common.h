#ifndef __CS267_COMMON_H__
#define __CS267_COMMON_H__

inline int mymin( int a, int b ) { return a < b ? a : b; }
inline int mymax( int a, int b ) { return a > b ? a : b; }

#define density 0.0005
#define mass    0.01
#define cutoff  0.01
#define min_r   (cutoff/100)
#define dt      0.0005

#define MAX_BIN_SIZE 5 // Should be 4 but tbh I'm not 100% sure so... 5


//
//  saving parameters
//
const int NSTEPS = 1000;
const int SAVEFREQ = 10;

//
// particle data structure
//
typedef struct 
{
  int binNum;
  double x;
  double y;
  double vx;
  double vy;
  double ax;
  double ay;
} particle_t;

//
// structure of arrays
//
typedef struct 
{
  double * x;
  double * y;
  double * vx;
  double * vy;
  double * ax;
  double * ay;
} particle_arr_t;

//
// structure of arrays
//
typedef struct 
{
  int size;
  particle_t * arr[5];
} bin_t;

//
//  timing routines
//
double read_timer( );

//
//  simulation routines
//

void set_size( int n );
int get_bin_row_size();
void init_particles( int n, particle_t *p );
void init_particles_array( int n, particle_arr_t &p );
void apply_force( particle_t &particle, particle_t &neighbor );
void move( particle_t &p );

//
//  I/O routines
//
FILE *open_save( char *filename, int n );
void save( FILE *f, int n, particle_t *p );

//
//  argument processing routines
//
int find_option( int argc, char **argv, const char *option );
int read_int( int argc, char **argv, const char *option, int default_value );
char *read_string( int argc, char **argv, const char *option, char *default_value );

#endif
