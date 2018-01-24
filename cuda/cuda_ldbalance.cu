#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include "bitmap_image.hpp"

#define MazesizeX 500
#define MazesizeY 300
#define vert MazesizeX*MazesizeY

#define WHITE 1
#define BLACK 2
#define GRAY  3

#define UP 1
#define DOWN 2
#define LEFT 3
#define RIGHT 4

#define blockNum  20
#define threadNum 50
#define cores blockNum*threadNum

using namespace std;
__device__ volatile int cuda_restnum = vert;

__device__ volatile bool cuda_relocate_list[cores] = {false};

__device__ volatile int signal[cores] = {0};
__device__ volatile bool signal_valid[cores] = {false};


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
	// printf("thread %d cand_num = %d\n", thread_index, cand_num);
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
		// printf("no direction, next_node %d\n", next_node);
		*next_direction = -1;
	}
	return next_node;
}

__device__
void mergetree(int treei, int treej, int *cuda_colormap, int *cuda_status) { // merge treej into treei
	// merge tree node
	// printf("MERGE: tree:%d and tree:%d \n", treei, treej);
	// if(treej < 0 || treei < 0) return;
	for(int i = 0; i < vert; i++) {
		if(cuda_status[treej*vert + i] == BLACK) {
				printf("tree %d, %d is black\n",treej, i);
				cuda_status[treei* vert + i] = BLACK;
			}
		if(cuda_status[treej* vert + i] == GRAY)
			if(cuda_status[treei * vert + i] == WHITE)
				cuda_status[treei * vert + i] = GRAY;
		if(cuda_colormap[i] == treej) cuda_colormap[i] = treei;
		cuda_status[treej * vert + i] = WHITE;
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


__device__
int thread_relocate(int thread_index, int* cuda_colormap, int* cuda_status) {
	int order = -1, temp = 0;
	int next_node = -1;	
	for(int i = 0; i < cores; i++) {
		if(cuda_relocate_list[i]) {
			order++;
			if(i == thread_index) {
				break;		
			}
		}
	}
	if(order == -1 || !cuda_relocate_list[thread_index]) return -1;
	for(int i = 0; i < vert; i++) {
		if(cuda_colormap[i] == -1) {
			if(temp == order) {		

				cuda_status[vert * thread_index + i] = BLACK;
				if(i >= MazesizeX)
					cuda_status[thread_index * vert + i - MazesizeX] = GRAY;
				if(i + MazesizeX < vert)
					cuda_status[thread_index * vert + i + MazesizeX] = GRAY;
				if(i % MazesizeX > 0) 
					cuda_status[thread_index * vert + i - 1] = GRAY;
				if(i % MazesizeX < (MazesizeX - 1)) 
					cuda_status[thread_index * vert + i + 1] = GRAY;
				cuda_colormap[i] = thread_index;
				next_node = i;
				break;
			}
			else temp++;
		}
	}
	if(next_node >= 0){
		printf("thread %d restarted by signal at %d, color:%d\n", thread_index, next_node, cuda_colormap[next_node]);		
		cuda_relocate_list[order] = false;
		
	}

	return next_node;
}




__global__
void MSTthread(int *cuda_map, int *cuda_colormap, int *cuda_status) {
	int thread_index = threadIdx.x + blockDim.x * blockIdx.x;
	int next_node, next_direction;
	volatile int signal_recved;
	int relocate_node;
	bool flag;
	// printf("thread %d starts\n", thread_index);
	signal[thread_index] = 0;
	signal_valid[thread_index] = false;
	while(1) {
		//if(share * thread_index >= vert) return;
		flag = true;
		while(flag) {
			signal_recved = recv_signal(thread_index, signal, signal_valid); 
			
			if( signal_recved != -1 && signal_recved < thread_index) {
				// return;				
				// here I'll implement a new way to "restart" the thread
				relocate_node = thread_relocate(thread_index, cuda_colormap, cuda_status);
				if(relocate_node == -1) {
					printf("thread %d finally no pos\n", thread_index);
					return;
				}
				else {
					
					// continue;
				}
			}
			next_node = -1;
			next_node = findnext(cuda_colormap, cuda_status, thread_index, &next_direction);
			
			if(next_node == -1) {
				printf("thread %d return for no node avaliable\n", thread_index);			
				return;
			}
			else { 
				if(thread_index == 0) {printf("thread 0 finds %d, color: %d\n", next_node, cuda_colormap[next_node]);}
				if(cuda_colormap[next_node] == -1) {
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
				}
				
				else if(cuda_colormap[next_node] != thread_index) {
					int target_color = cuda_colormap[next_node];
					if(target_color < thread_index) {
						signal[thread_index] = target_color;
						signal_valid[thread_index] = true;
						signal[target_color] = thread_index;
						signal_valid[target_color] = true;
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
						printf("thread %d get merged at %d\n", thread_index, next_node);
						// add the 
						cuda_relocate_list[thread_index] = true;
						//thread_relocate(thread_index, cuda_colormap, cuda_status);
						
												
						// return;
					}
					else if(target_color > thread_index) {
						signal[thread_index] = target_color;
						signal_valid[thread_index] = true;
						signal[target_color] = thread_index;
						signal_valid[target_color] = true;
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
					}
					else {
						// unlock minnode
						continue;
					}
				}
				else if(cuda_colormap[next_node] == thread_index) {
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
					continue;
				}			
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
	cudaMemcpy(cuda_status, status, vert*cores*sizeof(int), cudaMemcpyHostToDevice);
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
	cudaMemcpy(colormap, cuda_colormap, vert*sizeof(int), cudaMemcpyDeviceToHost);
	print(map);
	//print_ascii(map);
	cudaFree(cuda_map);
	cudaFree(cuda_colormap);
	cudaFree(cuda_status);
	/*
	for(int i = 0; i < vert; i++) {
		cout << colormap[i] << ' ';
		if((i +1)% 20 ==  0) 
			cout << endl;
	}
	*/
	return 0;
}












