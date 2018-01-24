#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include "bitmap_image.hpp"

#define MazesizeX 200
#define MazesizeY 100
#define vert MazesizeX*MazesizeY

#define WHITE 1
#define BLACK 2
#define GRAY  3

#define UP 1
#define DOWN 2
#define LEFT 3
#define RIGHT 4

#define blockNum  10
#define threadNum 100
#define cores blockNum*threadNum

using namespace std;
__device__ volatile int cuda_restnum = vert;
__device__ volatile int find_next_sem = 0;
__device__ volatile int signal_sem = 0;
__device__ volatile int minnode_sem = 0;
__device__ volatile int signal[cores] = {0};
__device__ volatile bool signal_valid[cores] = {false};

__device__ void acquire_semaphore(volatile int *lock, int source_thread){
  	while (1) {
		if(atomicCAS((int *)lock, 0, 1) != 0)
			break;
	}
  	//printf("locked by %d thread\n", source_thread);

}

__device__ void release_semaphore(volatile int *lock, int source_thread){
	  //printf("unlocked by %d thread\n", source_thread);
	  atomicExch( (int *) lock, 0 );
 }

__device__ int findnext(int *cuda_colormap, int *cuda_status, 
			int thread_index, int* next_direction) {
	int candidate[vert], cand_num = 0;
	int pot_direction[4], dir_num = 0;
	int next_node;
	for(int i = 0; i < vert; i++) {
		if(cuda_status[thread_index * vert + i] == GRAY) {
			candidate[cand_num++] = i; 
		}
	}
	// randomly select a possible node as candidate
	curandState_t candidate_state;
	curand_init(thread_index, 0, cuda_restnum, &candidate_state);
	 printf("thread %d cand_num = %d\n", thread_index, cand_num);
	if(!cand_num) return -1;
	next_node = candidate[curand(&candidate_state)%cand_num];
	// printf("next node: %d, dir_num = %d\n", next_node, dir_num);
	// randomly select a direction
	if(next_node + MazesizeX < vert)
		if(cuda_status[thread_index * vert + next_node + MazesizeX] == BLACK){
			pot_direction[dir_num++] = UP;
			// printf("thread %d, nextnode %d, UP\n", thread_index, next_node);
		}
	if(next_node >= MazesizeX)
		if(cuda_status[thread_index * vert + next_node - MazesizeX] == BLACK){
			pot_direction[dir_num++] = DOWN;
			// printf("thread %d, nextnode %d, DOWN\n", thread_index, next_node);
		}
	if(next_node % MazesizeX != 0)
		if(cuda_status[thread_index * vert + next_node - 1] == BLACK){
			pot_direction[dir_num++] = LEFT;
			// printf("thread %d, nextnode %d, LEFT\n", thread_index, next_node);
		}
	if(next_node % MazesizeX != MazesizeX - 1)
		if(cuda_status[thread_index * vert + next_node + 1] == BLACK){
			pot_direction[dir_num++] = RIGHT;
			// printf("thread %d, nextnode %d, RIGHT\n", thread_index, next_node);
		}

	curandState_t direction_state;
	curand_init(thread_index, 0, cuda_restnum, &direction_state);
	
	if(dir_num)	
		*next_direction =  pot_direction[curand(&direction_state)%dir_num];
	else {
		printf("no direction, next_node %d\n", next_node);
		*next_direction = -1;
	}
	return next_node;
}

__device__
void mergetree(int treei, int treej, int *cuda_colormap, int *cuda_status) { // merge treej into treei
	// merge tree node
	printf("MERGE: tree:%d and tree:%d \n", treei, treej);
	for(int i = 0; i < vert; i++) {
		if(cuda_status[treej*vert + i] == BLACK)
			cuda_status[treei* vert + i] = BLACK;
		if(cuda_status[treej* vert + i] == GRAY)
			if(cuda_status[treei * vert + i] == WHITE)
				cuda_status[treei * vert + i] = GRAY;
	}
}

__device__
int recv_signal(int local_index, volatile int signal_map[cores], volatile bool signal_valid[cores]) {
	if(signal_valid[local_index]) {
		signal_valid[local_index] = false;
		return signal_map[local_index];
	}
	else return -1;
}


__global__
void MSTthread(int *cuda_map, int *cuda_colormap, int *cuda_status) {
	int thread_index = threadIdx.x + blockDim.x * blockIdx.x;
	int next_node, next_direction;
	volatile int signal_recved;
	bool flag;
	printf("thread %d starts\n", thread_index);
	signal[thread_index] = 0;
	signal_valid[thread_index] = false;
	while(1) {
		
		if(cuda_restnum == 0) {
			printf("all nodes found\n");			
			return;
		}
		flag = true;
		while(flag) {
			signal_recved = recv_signal(thread_index, signal, signal_valid); 
			if( signal_recved != -1 && signal_recved < thread_index) {
				printf("thread %d terminated by signal\n", thread_index);
				return;
			}
			
			next_node = findnext(cuda_colormap, cuda_status, thread_index, &next_direction);
			printf("thread %d find next node %d\n", thread_index, next_node);
			if(next_node == -1) {
				printf("thread %d return for no node avaliable\n", thread_index);			
				return;
			}
			else { 
				acquire_semaphore(&minnode_sem, thread_index);
				if(cuda_colormap[next_node] == -1) {
					acquire_semaphore(&signal_sem, thread_index);
					cuda_colormap[next_node] = thread_index;
					switch(next_direction) {
						case UP:
							cuda_map[next_node * 4] = 1;
							cuda_map[(next_node + MazesizeX) * 4 + 1] = 1;
							break;
						case DOWN:
							cuda_map[next_node * 4 + 1] = 1;
							cuda_map[(next_node-MazesizeX) * 4] = 1;
							break;
						case LEFT:
							cuda_map[next_node * 4 + 2] = 1;
							cuda_map[(next_node - 1) * 4 + 3] = 1;
							break;
						case RIGHT:
							cuda_map[next_node * 4 + 3] = 1;
							cuda_map[(next_node + 1) * 4 + 2] = 1;
							break;
						default: break;
						
					}
					
					release_semaphore(&minnode_sem, thread_index);
					// update status and keyarray and cuda_restnum					
					cuda_status[thread_index * vert + next_node] = BLACK;
					if(next_node >= MazesizeX)
						if(cuda_status[thread_index * vert +next_node - MazesizeX] == WHITE)
							cuda_status[thread_index * vert + next_node - MazesizeX] = GRAY;
					if(next_node + MazesizeX < vert)
						if(cuda_status[thread_index * vert + next_node + MazesizeX] == WHITE)
							cuda_status[thread_index * vert + next_node + MazesizeX] = GRAY;
					if(next_node % MazesizeX > 0) 
						if(cuda_status[thread_index * vert + next_node - 1] == WHITE)
							cuda_status[thread_index * vert + next_node - 1] = GRAY;
					if(next_node % MazesizeX < (MazesizeX - 1)) 
						if(cuda_status[thread_index * vert + next_node + 1] == WHITE)
							cuda_status[thread_index * vert + next_node + 1] = GRAY;
					//cuda_restnum--;
					//printf("----%d left----\n", cuda_restnum);
					release_semaphore(&signal_sem, thread_index);
				}
				
				else if(cuda_colormap[next_node] != thread_index) {
					int target_color = cuda_colormap[next_node];
					if(target_color < thread_index) {
						signal[thread_index] = target_color;
						signal_valid[thread_index] = true;
						//while(!signal_valid[target_color] ){};
						mergetree(target_color, thread_index, cuda_colormap, cuda_status);
						switch(next_direction) {
							case UP:
								cuda_map[next_node * 4] = 1;
								cuda_map[(next_node + MazesizeX) * 4 + 1] = 1;
								break;
							case DOWN:
								cuda_map[next_node * 4 + 1] = 1;
								cuda_map[(next_node-MazesizeX) * 4] = 1;
								break;
							case LEFT:
								cuda_map[next_node * 4 + 2] = 1;
								cuda_map[(next_node - 1) * 4 + 3] = 1;
								break;
							case RIGHT:
								cuda_map[next_node * 4 + 3] = 1;
								cuda_map[(next_node + 1) * 4 + 2] = 1;
								break;
							default: break;
						
						}
						// unlock minnode and kill j
						release_semaphore(&minnode_sem, thread_index);
						printf("thread %d get merged\n", thread_index);
						return;
					}
					else if(target_color > thread_index) {
						signal[thread_index] = target_color;
						signal_valid[thread_index] = true;
						//while(!signal_valid[target_color]) {};
						mergetree(thread_index, target_color, cuda_colormap, cuda_status);
						switch(next_direction) {
							case UP:
								cuda_map[next_node * 4] = 1;
								cuda_map[(next_node + MazesizeX) * 4 + 1] = 1;
								break;
							case DOWN:
								cuda_map[next_node * 4 + 1] = 1;
								cuda_map[(next_node-MazesizeX) * 4] = 1;
								break;
							case LEFT:
								cuda_map[next_node * 4 + 2] = 1;
								cuda_map[(next_node - 1) * 4 + 3] = 1;
								break;
							case RIGHT:
								cuda_map[next_node * 4 + 3] = 1;
								cuda_map[(next_node + 1) * 4 + 2] = 1;
								break;
							default: break;
						
						}
						// unlock minnode and kill j
						release_semaphore(&minnode_sem, thread_index);
					}
					else {
						// unlock minnode
						release_semaphore(&minnode_sem, thread_index);
						continue;
					}
				}
				else if(cuda_colormap[next_node] == thread_index)
					continue;
			
			}
		}
	}
}




void print(int map[vert * 4]) {
	int dimx = MazesizeX*10;
	int dimy = MazesizeY*10;
	bitmap_image image(dimx+1, dimy+1);
	for(int x = 0; x < dimx; x++)
		for(int y = 1; y < dimy; y++) {
		if(x%10 == 0 || y %10 == 0) {
			if(x%10 == 0) {
				if(map[((y/10)*MazesizeX+(x/10))* 4 + 2]) image.set_pixel(x,dimy+1-y,255,255,255);			
				else image.set_pixel(x,dimy+1-y,0,0,0);
			}
			if(y%10 == 0) {
				if(map[((y/10)*MazesizeX+(x/10))* 4 + 1]) image.set_pixel(x,dimy+1-y,255,255,255);			
				else image.set_pixel(x,dimy+1-y,0,0,0);
			}
		}
		else image.set_pixel(x,dimy+1-y,255,255,255);
		}	
	image.save_image("maze.bmp");
}

void print_ascii(int map[vert * 4]) {
	for(int j = 0; j < MazesizeX; j++) {
		cout << "--";
		}
	cout << "-" << endl;
	for(int i = MazesizeY-1; i >= 0; i--) {
		for(int j = 0; j < MazesizeX; j++) {
			if(!map[(i*MazesizeX+j)* 4 + 2]) cout << '|';
			else cout << ' ';
			cout << ' ';					
		}
		cout << '|' << endl;
		for(int j = 0; j < MazesizeX; j++) {
			if(!map[(i*MazesizeX+j)* 4 + 1]) cout << "--";
			else cout << "- ";
		}
		cout << '-' << endl;
	}	
	return;
}

int main(int argc, char** argv) {
	// declare data structure 
	int *map, *cuda_map;
	int *colormap, *cuda_colormap;
	int *status, *cuda_status;


	// initialization
	map = new int[vert * 4];
	if(cudaSuccess != cudaMalloc(&cuda_map, vert * 4 * sizeof(int))) {
		printf("map create failed\n");
		return -1;
	}
	colormap = new int[vert];
	if(cudaSuccess != cudaMalloc(&cuda_colormap, vert * sizeof(int))) {
		printf("color map create failed\n");
		return -1;
	}
	status = new int[vert * cores];
	if(cudaSuccess != cudaMalloc(&cuda_status, vert * cores * sizeof(int)))	{
		printf("status map create failed\n");	
		return -1;
	}
	
	for(int i = 0; i < vert * 4; i++) {
		map[i] = 0;	
	}
	for(int i = 0; i < vert; i++) { 
		colormap[i] = -1;
	}
	int share = vert / (cores) * (cores) == vert?
			vert/(cores) : vert/(cores)+1;
	cout << "share = " << share << endl;
	for(int i = 0; i < cores; i++) {
		for(int j = 0; j < vert; j++) {
			if(j == share * i) status[i*vert+j] = GRAY;
			else status[i*vert+j] = WHITE;
		}
	}
	cudaMemcpy(cuda_map, map, vert*4*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_colormap, colormap, vert*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_status, status, vert*blockNum*threadNum*sizeof(int), cudaMemcpyHostToDevice);
	// start timing
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaEventRecord(start);


	// start

	MSTthread <<< blockNum, threadNum >>>(cuda_map, cuda_colormap, cuda_status);
	cudaDeviceSynchronize();

	cudaEventRecord(stop);
	
	cudaEventSynchronize(stop);
	
	float time = 0;
	cudaEventElapsedTime(&time, start, stop);
	cout << "Total time = " << time << "ms" << endl;
	// print 
	cudaMemcpy(map, cuda_map, vert*4*sizeof(int), cudaMemcpyDeviceToHost);
	print(map);
	//print_ascii(map);
	cudaFree(cuda_map);
	cudaFree(cuda_colormap);
	cudaFree(cuda_status);
	return 0;
}












