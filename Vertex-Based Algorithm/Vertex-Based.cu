/*************************************************************************************************************************************************
Implementing Single Source Shortest Path given in TTCS paper "Locality-Based Relaxation: An Efficient Method for GPU-Based Computation of Shortest Paths", 2017.

Created by Mohsen Safari.
**************************************************************************************************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
//#include <cutil.h>

#define MAX_THREADS_PER_BLOCK 256
#define MAX_COST 10000000

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//CUDA Kernels
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void DijkastraKernel1(int* g_graph_nodes, int* g_graph_edges,int* g_graph_weights, bool* g_graph_mask1, bool* g_graph_mask2 , int* g_cost , int no_of_nodes, int edge_list_size, bool *d_finished)
{
	int tid = (blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x);	
	int tid1 = 2*(blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x);
	int tid2 = 2*(blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x)+1;
	if(tid == (no_of_nodes/2))
		tid2 = no_of_nodes - 1;
	if(tid2<no_of_nodes)
	{
		int i, j, k, l, edge_reg_i, edge_reg_j, edge_reg_k, edge_reg_l, end, end1, end2, end3;
		end = end1 = end2 = end3 = edge_list_size;
		if(g_graph_mask1[tid1])
		{
			g_graph_mask1[tid1] = false;
			if(tid1 != no_of_nodes-1)
				end = g_graph_nodes[tid1+1];
			for(i = g_graph_nodes[tid1]; i<end; i++)
			{
				edge_reg_i = g_graph_edges[i];
       				if(g_cost[tid1]+g_graph_weights[i] < g_cost[edge_reg_i])
        			{
				
	  				atomicMin(&g_cost[edge_reg_i], g_cost[tid1]+g_graph_weights[i]);
					if (g_graph_nodes[i] != no_of_nodes-1)
						end1 = g_graph_nodes[edge_reg_i+1];
					for(j = g_graph_nodes[edge_reg_i]; j<end1; j++)
					{
						edge_reg_j = g_graph_edges[j];
						if(g_cost[edge_reg_i]+g_graph_weights[j] < g_cost[edge_reg_j])
						{
							atomicMin(&g_cost[edge_reg_j], g_cost[edge_reg_i]+g_graph_weights[j]);
							if (g_graph_nodes[j] != no_of_nodes-1)
								end2 = g_graph_nodes[edge_reg_j+1];
							for(k = g_graph_nodes[edge_reg_j]; k<end2; k++)
							{
								edge_reg_k = g_graph_edges[k];
								if(g_cost[edge_reg_j]+g_graph_weights[k] < g_cost[edge_reg_k])
								{
									atomicMin(&g_cost[edge_reg_k], g_cost[edge_reg_j]+g_graph_weights[k]);	
									if (g_graph_nodes[k] != no_of_nodes-1)
										end3 = g_graph_nodes[edge_reg_k+1];
									for(l = g_graph_nodes[edge_reg_k]; l<end3; l++)
									{
										edge_reg_l = g_graph_edges[l];
										if(g_cost[edge_reg_k]+g_graph_weights[l] < g_cost[edge_reg_l])
										{
											atomicMin(&g_cost[edge_reg_l], g_cost[edge_reg_k]+g_graph_weights[l]);	
											g_graph_mask2[edge_reg_l] = true;
        	  									*d_finished = true;	
										}	
									}
								}
							}
						}
					}
          			
			
	    			}
			}
		}
		if(g_graph_mask1[tid2])
		{	
			g_graph_mask1[tid2] = false;
			end = g_graph_nodes[tid2+1];
			if(tid2 == no_of_nodes-1)
				end = edge_list_size;	
			for(i = g_graph_nodes[tid2]; i<end; i++)
			{
				edge_reg_i = g_graph_edges[i];
        			if(g_cost[tid2]+g_graph_weights[i] < g_cost[edge_reg_i])
        			{
	  				atomicMin(&g_cost[edge_reg_i], g_cost[tid2]+g_graph_weights[i]);
					end1 = edge_list_size;
          				if (g_graph_nodes[i] != no_of_nodes-1)
						end1 = g_graph_nodes[edge_reg_i+1];
					for(j = g_graph_nodes[edge_reg_i]; j<end1; j++)
					{
						edge_reg_j = g_graph_edges[j];
						if(g_cost[edge_reg_i]+g_graph_weights[j] < g_cost[edge_reg_j])
						{
							atomicMin(&g_cost[edge_reg_j], g_cost[edge_reg_i]+g_graph_weights[j]);
							end2 = edge_list_size;
							if (g_graph_nodes[j] != no_of_nodes-1)
								end2 = g_graph_nodes[edge_reg_j+1];
							for(k = g_graph_nodes[edge_reg_j]; k<end2; k++)
							{
								edge_reg_k = g_graph_edges[k];
								if(g_cost[edge_reg_j]+g_graph_weights[k] < g_cost[edge_reg_k])
								{
									atomicMin(&g_cost[edge_reg_k], g_cost[edge_reg_j]+g_graph_weights[k]);
									end3 = edge_list_size;
									if (g_graph_nodes[k] != no_of_nodes-1)
										end3 = g_graph_nodes[edge_reg_k+1];
									for(l = g_graph_nodes[edge_reg_k]; l<end3; l++)
									{
										edge_reg_l = g_graph_edges[l];
										if(g_cost[edge_reg_k]+g_graph_weights[l] < g_cost[edge_reg_l])
										{
											atomicMin(&g_cost[edge_reg_l], g_cost[edge_reg_k]+g_graph_weights[l]);	
											g_graph_mask2[edge_reg_l] = true;
        	  									*d_finished = true;	
										}	
									}
								}
							}	
						}
					}
				
				}
	    		} 
	

		}
			
 	 }

}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void DijkastraKernel2(int* g_graph_nodes, int* g_graph_edges,int* g_graph_weights, bool* g_graph_mask1, bool* g_graph_mask2 , int* g_cost , int no_of_nodes, int edge_list_size, bool *d_finished)
{
	int tid = (blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x);	
	int tid1 = 2*(blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x);
	int tid2 = 2*(blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x)+1;
	if(tid == (no_of_nodes/2))
		tid2 = no_of_nodes - 1;
	if(tid2<no_of_nodes)
	{
		int i, j, k, l, edge_reg_i, edge_reg_j, edge_reg_k, edge_reg_l, end, end1, end2, end3;
		end = end1 = end2 = end3 = edge_list_size;
		if(g_graph_mask2[tid1])
		{
			g_graph_mask2[tid1] = false;
			if(tid1 != no_of_nodes-1)
				end = g_graph_nodes[tid1+1];
			for(i = g_graph_nodes[tid1]; i<end; i++)
			{
				edge_reg_i = g_graph_edges[i];
        			if(g_cost[tid1]+g_graph_weights[i] < g_cost[edge_reg_i])
        			{
				
	  				atomicMin(&g_cost[edge_reg_i], g_cost[tid1]+g_graph_weights[i]);
					if (g_graph_nodes[i] != no_of_nodes-1)
						end1 = g_graph_nodes[edge_reg_i+1];
					for(j = g_graph_nodes[edge_reg_i]; j<end1; j++)
					{
						edge_reg_j = g_graph_edges[j];
						if(g_cost[edge_reg_i]+g_graph_weights[j] < g_cost[edge_reg_j])
						{
							atomicMin(&g_cost[edge_reg_j], g_cost[edge_reg_i]+g_graph_weights[j]);
							if (g_graph_nodes[j] != no_of_nodes-1)
								end2 = g_graph_nodes[edge_reg_j+1];
							for(k = g_graph_nodes[edge_reg_j]; k<end2; k++)
							{
								edge_reg_k = g_graph_edges[k];
								if(g_cost[edge_reg_j]+g_graph_weights[k] < g_cost[edge_reg_k])
								{
									atomicMin(&g_cost[edge_reg_k], g_cost[edge_reg_j]+g_graph_weights[k]);	
									if (g_graph_nodes[k] != no_of_nodes-1)
										end3 = g_graph_nodes[edge_reg_k+1];
									for(l = g_graph_nodes[edge_reg_k]; l<end3; l++)
									{
										edge_reg_l = g_graph_edges[l];
										if(g_cost[edge_reg_k]+g_graph_weights[l] < g_cost[edge_reg_l])
										{
											atomicMin(&g_cost[edge_reg_l], g_cost[edge_reg_k]+g_graph_weights[l]);	
											g_graph_mask1[edge_reg_l] = true;
        	  									*d_finished = true;	
										}	
									}	
								}
							}
						}
					}
          			
			
	    			}
			}
		}
		if(g_graph_mask2[tid2])
		{	
			g_graph_mask2[tid2] = false;
			end = g_graph_nodes[tid2+1];
			if(tid2 == no_of_nodes-1)
				end = edge_list_size;	
			for(i = g_graph_nodes[tid2]; i<end; i++)
			{
				edge_reg_i = g_graph_edges[i];
        			if(g_cost[tid2]+g_graph_weights[i] < g_cost[edge_reg_i])
        			{
	  				atomicMin(&g_cost[edge_reg_i], g_cost[tid2]+g_graph_weights[i]);
					end1 = edge_list_size;
          				if (g_graph_nodes[i] != no_of_nodes-1)
						end1 = g_graph_nodes[edge_reg_i+1];
					for(j = g_graph_nodes[edge_reg_i]; j<end1; j++)
					{
						edge_reg_j = g_graph_edges[j];
						if(g_cost[edge_reg_i]+g_graph_weights[j] < g_cost[edge_reg_j])
						{
							atomicMin(&g_cost[edge_reg_j], g_cost[edge_reg_i]+g_graph_weights[j]);
							end2 = edge_list_size;
							if (g_graph_nodes[j] != no_of_nodes-1)
								end2 = g_graph_nodes[edge_reg_j+1];
							for(k = g_graph_nodes[edge_reg_j]; k<end2; k++)
							{
								edge_reg_k = g_graph_edges[k];
								if(g_cost[edge_reg_j]+g_graph_weights[k] < g_cost[edge_reg_k])
								{
									atomicMin(&g_cost[edge_reg_k], g_cost[edge_reg_j]+g_graph_weights[k]);
									end3 = edge_list_size;
									if (g_graph_nodes[k] != no_of_nodes-1)
										end3 = g_graph_nodes[edge_reg_k+1];
									for(l = g_graph_nodes[edge_reg_k]; l<end3; l++)
									{
										edge_reg_l = g_graph_edges[l];
										if(g_cost[edge_reg_k]+g_graph_weights[l] < g_cost[edge_reg_l])
										{
											atomicMin(&g_cost[edge_reg_l], g_cost[edge_reg_k]+g_graph_weights[l]);	
											g_graph_mask1[edge_reg_l] = true;
        	  									*d_finished = true;	
										}	
									}
								}
							}	
						}
					}
				}
	    		}
		}
			
 	 }

}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void DijkastraKernel3(int* g_graph_nodes, int* g_graph_edges,int* g_graph_weights, bool* g_graph_mask1, bool* g_graph_mask2 , int* g_cost , int no_of_nodes, int edge_list_size, bool *d_finished)
{
	int tid = (blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x);	
	int tid1 = 2*(blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x);
	int tid2 = 2*(blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x)+1;
	if(tid == (no_of_nodes/2))
		tid2 = no_of_nodes - 1;
	if(tid2<no_of_nodes)
	{
		int i, j, k, l, edge_reg_i, edge_reg_j, edge_reg_k, edge_reg_l, end, end1, end2, end3;
		end = end1 = end2 = end3 = edge_list_size;
		if(g_graph_mask1[tid1])
		{
			g_graph_mask1[tid1] = false;
			if(tid1 != no_of_nodes-1)
				end = g_graph_nodes[tid1+1];
			for(i = g_graph_nodes[tid1]; i<end; i++)
			{
				edge_reg_i = g_graph_edges[i];
        			if(g_cost[tid1]+g_graph_weights[i] < g_cost[edge_reg_i])
        			{
				
	  				atomicMin(&g_cost[edge_reg_i], g_cost[tid1]+g_graph_weights[i]);
					if (g_graph_nodes[i] != no_of_nodes-1)
						end1 = g_graph_nodes[edge_reg_i+1];
					for(j = g_graph_nodes[edge_reg_i]; j<end1; j++)
					{
						edge_reg_j = g_graph_edges[j];
						if(g_cost[edge_reg_i]+g_graph_weights[j] < g_cost[edge_reg_j])
						{
							atomicMin(&g_cost[edge_reg_j], g_cost[edge_reg_i]+g_graph_weights[j]);
							if (g_graph_nodes[j] != no_of_nodes-1)
								end2 = g_graph_nodes[edge_reg_j+1];
							for(k = g_graph_nodes[edge_reg_j]; k<end2; k++)
							{
								edge_reg_k = g_graph_edges[k];
								if(g_cost[edge_reg_j]+g_graph_weights[k] < g_cost[edge_reg_k])
								{
									atomicMin(&g_cost[edge_reg_k], g_cost[edge_reg_j]+g_graph_weights[k]);	
									if (g_graph_nodes[k] != no_of_nodes-1)
										end3 = g_graph_nodes[edge_reg_k+1];
									for(l = g_graph_nodes[edge_reg_k]; l<end3; l++)
									{
										edge_reg_l = g_graph_edges[l];
										if(g_cost[edge_reg_k]+g_graph_weights[l] < g_cost[edge_reg_l])
										{
											atomicMin(&g_cost[edge_reg_l], g_cost[edge_reg_k]+g_graph_weights[l]);	
											g_graph_mask2[edge_reg_l] = true;
										}	
									}
								}
							}
						}
					}
          			
			
	    			}
			}
		}
		if(g_graph_mask1[tid2])
		{	
			g_graph_mask1[tid2] = false;
			end = g_graph_nodes[tid2+1];
			if(tid2 == no_of_nodes-1)
				end = edge_list_size;	
			for(i = g_graph_nodes[tid2]; i<end; i++)
			{
				edge_reg_i = g_graph_edges[i];
        			if(g_cost[tid2]+g_graph_weights[i] < g_cost[edge_reg_i])
        			{
	  				atomicMin(&g_cost[edge_reg_i], g_cost[tid2]+g_graph_weights[i]);
					end1 = edge_list_size;
          				if (g_graph_nodes[i] != no_of_nodes-1)
						end1 = g_graph_nodes[edge_reg_i+1];
					for(j = g_graph_nodes[edge_reg_i]; j<end1; j++)
					{
						edge_reg_j = g_graph_edges[j];
						if(g_cost[edge_reg_i]+g_graph_weights[j] < g_cost[edge_reg_j])
						{
							atomicMin(&g_cost[edge_reg_j], g_cost[edge_reg_i]+g_graph_weights[j]);
							end2 = edge_list_size;
							if (g_graph_nodes[j] != no_of_nodes-1)
								end2 = g_graph_nodes[edge_reg_j+1];
							for(k = g_graph_nodes[edge_reg_j]; k<end2; k++)
							{
								edge_reg_k = g_graph_edges[k];
								if(g_cost[edge_reg_j]+g_graph_weights[k] < g_cost[edge_reg_k])
								{
									atomicMin(&g_cost[edge_reg_k], g_cost[edge_reg_j]+g_graph_weights[k]);
									end3 = edge_list_size;
									if (g_graph_nodes[k] != no_of_nodes-1)
										end3 = g_graph_nodes[edge_reg_k+1];
									for(l = g_graph_nodes[edge_reg_k]; l<end3; l++)
									{
										edge_reg_l = g_graph_edges[l];
										if(g_cost[edge_reg_k]+g_graph_weights[l] < g_cost[edge_reg_l])
										{
											atomicMin(&g_cost[edge_reg_l], g_cost[edge_reg_k]+g_graph_weights[l]);	
											g_graph_mask2[edge_reg_l] = true;	
										}	
									}
								}
							}	
						}
					}
				}
	    		}
		}
			
  	}

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void DijkastraKernel4(int* g_graph_nodes, int* g_graph_edges,int* g_graph_weights, bool* g_graph_mask1, bool* g_graph_mask2 , int* g_cost , int no_of_nodes, int edge_list_size, bool *d_finished)
{
	int tid = (blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x);	
	int tid1 = 2*(blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x);
	int tid2 = 2*(blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x)+1;
	if(tid == (no_of_nodes/2))
		tid2 = no_of_nodes - 1;
	if(tid2<no_of_nodes)
	{
		int i, j, k, l, edge_reg_i, edge_reg_j, edge_reg_k, edge_reg_l, end, end1, end2, end3;
		end = end1 = end2 = end3 = edge_list_size;
		if(g_graph_mask2[tid1])
		{
			g_graph_mask2[tid1] = false;
			if(tid1 != no_of_nodes-1)
				end = g_graph_nodes[tid1+1];
			for(i = g_graph_nodes[tid1]; i<end; i++)
			{
				edge_reg_i = g_graph_edges[i];
        			if(g_cost[tid1]+g_graph_weights[i] < g_cost[edge_reg_i])
        			{
				
	  				atomicMin(&g_cost[edge_reg_i], g_cost[tid1]+g_graph_weights[i]);
					if (g_graph_nodes[i] != no_of_nodes-1)
						end1 = g_graph_nodes[edge_reg_i+1];
					for(j = g_graph_nodes[edge_reg_i]; j<end1; j++)
					{
						edge_reg_j = g_graph_edges[j];
						if(g_cost[edge_reg_i]+g_graph_weights[j] < g_cost[edge_reg_j])
						{
							atomicMin(&g_cost[edge_reg_j], g_cost[edge_reg_i]+g_graph_weights[j]);
							if (g_graph_nodes[j] != no_of_nodes-1)
								end2 = g_graph_nodes[edge_reg_j+1];
							for(k = g_graph_nodes[edge_reg_j]; k<end2; k++)
							{
								edge_reg_k = g_graph_edges[k];
								if(g_cost[edge_reg_j]+g_graph_weights[k] < g_cost[edge_reg_k])
								{
									atomicMin(&g_cost[edge_reg_k], g_cost[edge_reg_j]+g_graph_weights[k]);	
									if (g_graph_nodes[k] != no_of_nodes-1)
										end3 = g_graph_nodes[edge_reg_k+1];
									for(l = g_graph_nodes[edge_reg_k]; l<end3; l++)
									{
										edge_reg_l = g_graph_edges[l];
										if(g_cost[edge_reg_k]+g_graph_weights[l] < g_cost[edge_reg_l])
										{
											atomicMin(&g_cost[edge_reg_l], g_cost[edge_reg_k]+g_graph_weights[l]);	
											g_graph_mask1[edge_reg_l] = true;	
										}	
									}		
								}
							}
						}
					}
          			
			
	    			}
			}
		}
		if(g_graph_mask2[tid2])
		{	
			g_graph_mask2[tid2] = false;
			end = g_graph_nodes[tid2+1];
			if(tid2 == no_of_nodes-1)
				end = edge_list_size;	
			for(i = g_graph_nodes[tid2]; i<end; i++)
			{
				edge_reg_i = g_graph_edges[i];
        			if(g_cost[tid2]+g_graph_weights[i] < g_cost[edge_reg_i])
        			{
	  				atomicMin(&g_cost[edge_reg_i], g_cost[tid2]+g_graph_weights[i]);
					
          				if (g_graph_nodes[i] != no_of_nodes-1)
						end1 = g_graph_nodes[edge_reg_i+1];
					for(j = g_graph_nodes[edge_reg_i]; j<end1; j++)
					{
						edge_reg_j = g_graph_edges[j];
						if(g_cost[edge_reg_i]+g_graph_weights[j] < g_cost[edge_reg_j])
						{
							atomicMin(&g_cost[edge_reg_j], g_cost[edge_reg_i]+g_graph_weights[j]);
							end2 = edge_list_size;
							if (g_graph_nodes[j] != no_of_nodes-1)
								end2 = g_graph_nodes[edge_reg_j+1];
							for(k = g_graph_nodes[edge_reg_j]; k<end2; k++)
							{
								edge_reg_k = g_graph_edges[k];
								if(g_cost[edge_reg_j]+g_graph_weights[k] < g_cost[edge_reg_k])
								{
									atomicMin(&g_cost[edge_reg_k], g_cost[edge_reg_j]+g_graph_weights[k]);
									end3 = edge_list_size;
									if (g_graph_nodes[k] != no_of_nodes-1)
										end3 = g_graph_nodes[edge_reg_k+1];
									for(l = g_graph_nodes[edge_reg_k]; l<end3; l++)
									{
										edge_reg_l = g_graph_edges[l];
										if(g_cost[edge_reg_k]+g_graph_weights[l] < g_cost[edge_reg_l])
										{
											atomicMin(&g_cost[edge_reg_l], g_cost[edge_reg_k]+g_graph_weights[l]);	
											g_graph_mask1[edge_reg_l] = true;	
										}	
									}	
								}
							}	
						}
					}
				}
	    		}
		}
			
  	}

}

////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
	int repetition = 100;  // Repeat the algorithm for this number (of random sources) and take average time  
	float total_time=0;
	int query, queryarray[repetition];
	FILE *fp0;
	fp0 = fopen("/home/mohsen/Input/NewYorkQueries.txt", "r"); // The address of a query file to get sources from the input graph
	if(!fp0)
	{
		printf("Error reading query file\n");
		return 0;
	}
	for( unsigned int i = 0; i < repetition; i++) // Store the random sources from the query file
   	 {
		fscanf(fp0,"%d",&query);
        	queryarray[i] = query;
   	 }
	
	fclose(fp0); 
	
	for(int r=0; r<repetition; r++) // Repeat the program for a fix number of random sources and at the end take an average over time
	{	
		int no_of_nodes = 0;
		int edge_list_size = 0;
		FILE *fp;
		
		fp = fopen("/home/mohsen/Input/NewYork-CSR.txt", "r"); // The address of the input graph
		if(!fp)
		{
			printf("Error reading graph file\n");
			return 0;
		}
	
		int source = 0;
		
		fscanf(fp,"%d",&no_of_nodes);
		printf("Number of nodes: %d\n ",no_of_nodes);
	
		int num_of_blocks = 1;
		int num_of_threads_per_block = no_of_nodes;
	
		
		
		if(no_of_nodes>MAX_THREADS_PER_BLOCK) // Distribute threads across multiple blocks if necessary
		{
			num_of_blocks = (int)ceil(no_of_nodes/(double)(2*MAX_THREADS_PER_BLOCK)); 
			num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
		}
		
		// Allocate Host memory
    		int* h_graph_nodes = (int*) malloc(sizeof(int)*no_of_nodes);
    		bool *h_graph_mask1 = (bool*) malloc(sizeof(bool)*no_of_nodes);
    		bool *h_graph_mask2 = (bool*) malloc(sizeof(bool)*no_of_nodes);

    		int start, edgeno;   

   		// Initalize the memory
		int no = 0;
   		for( unsigned int i = 0; i < no_of_nodes; i++) 
   		{
			fscanf(fp,"%d %d",&start,&edgeno);
			if(edgeno>100)
				no++;
        		h_graph_nodes[i] = start;
        		h_graph_mask1[i] = false;
			h_graph_mask2[i] = false;
    		}
    
    		// Read the first source from the file
   		fscanf(fp,"%d",&source);
        
      		// Read and store edges and weights
    		fscanf(fp,"%d",&edge_list_size);
    		printf("Number of edges: %d\n",edge_list_size);    
    		int id;
    		int* h_graph_edges = (int*) malloc(sizeof(int)*edge_list_size);
    		int* h_graph_weights = (int*) malloc(sizeof(int)*edge_list_size);
    		for(int i=0; i < edge_list_size ; i++)
    		{
			fscanf(fp,"%d",&id);
			h_graph_edges[i] = id;
			fscanf(fp,"%d",&id);
			h_graph_weights[i] = id;
		
    		}
    
    
		if(fp)
			fclose(fp);    


		// Allocate and initialize the memory 
		int* h_cost = (int*) malloc( sizeof(int)*no_of_nodes);
		for(int i=0;i<no_of_nodes;i++)
			h_cost[i]= MAX_COST;	
	
		// Initialize the source
		source = queryarray[r]; 
        	h_cost[source] = 0;
        	h_graph_mask1[source] = true;
	
	
		// Copy arrays from Host to Device memory
    		int* d_graph_nodes;
    		cudaMalloc( (void**) &d_graph_nodes, sizeof(int)*no_of_nodes) ;
    		cudaMemcpy( d_graph_nodes, h_graph_nodes, sizeof(int)*no_of_nodes, cudaMemcpyHostToDevice) ;

		int* d_graph_edges;
    		cudaMalloc( (void**) &d_graph_edges, sizeof(int)*edge_list_size) ;
    		cudaMemcpy( d_graph_edges, h_graph_edges, sizeof(int)*edge_list_size, cudaMemcpyHostToDevice) ;

		int* d_graph_weights;
    		cudaMalloc( (void**) &d_graph_weights, sizeof(int)*edge_list_size) ;
    		cudaMemcpy( d_graph_weights, h_graph_weights, sizeof(int)*edge_list_size, cudaMemcpyHostToDevice) ;
    
    		bool* d_graph_mask1;
    		cudaMalloc( (void**) &d_graph_mask1, sizeof(bool)*no_of_nodes) ;
    		cudaMemcpy( d_graph_mask1, h_graph_mask1, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice) ;

		bool* d_graph_mask2;
    		cudaMalloc( (void**) &d_graph_mask2, sizeof(bool)*no_of_nodes) ;
    		cudaMemcpy( d_graph_mask2, h_graph_mask2, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice) ;
    
    		int* d_cost;
    		cudaMalloc( (void**) &d_cost, sizeof(int)*no_of_nodes);
    		cudaMemcpy( d_cost, h_cost, sizeof(int)*no_of_nodes, cudaMemcpyHostToDevice) ;

	
    		// A boolean to check the termination of the algorithm
		bool *d_finished;
		bool finished;
		cudaMalloc( (void**) &d_finished, sizeof(bool));
    
        	// Setup execution parameters
        	dim3  grid( num_of_blocks, 1, 1);
        	dim3  threads( num_of_threads_per_block, 1, 1);

		int* temp = (int*) malloc(sizeof(int)*no_of_nodes);
		int* SSSP = (int *)malloc((no_of_nodes+1) * sizeof(int *));
		int counter = 0;

		// Start the timer
  		cudaEvent_t begin, end;
		float time;
		cudaEventCreate(&begin);
		cudaEventCreate(&end);
		cudaEventRecord(begin, 0);

		// Kernel launches without CPU-GPU communication for a fix number (counter) for each graph
		do
		{
			DijkastraKernel3<<< grid, threads, 0 >>>( d_graph_nodes, d_graph_edges, d_graph_weights, d_graph_mask1, d_graph_mask2,    				d_cost, no_of_nodes, edge_list_size, d_finished);
			DijkastraKernel4<<< grid, threads, 0 >>>( d_graph_nodes, d_graph_edges, d_graph_weights, d_graph_mask1, d_graph_mask2, 				d_cost, no_of_nodes, edge_list_size, d_finished);
			counter++;
		
		}
		while(counter<=64);

		// Kernel launches with CPU-GPU communication via a boolean variable (finished)
		do
		{
			finished=false;
			cudaMemcpy( d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice) ;
			DijkastraKernel1<<< grid, threads, 0 >>>( d_graph_nodes, d_graph_edges, d_graph_weights, d_graph_mask1, d_graph_mask2, 				d_cost, no_of_nodes, edge_list_size, d_finished);
			DijkastraKernel2<<< grid, threads, 0 >>>( d_graph_nodes, d_graph_edges, d_graph_weights, d_graph_mask1, d_graph_mask2, 				d_cost, no_of_nodes, edge_list_size, d_finished);
			finished=false;
			cudaMemcpy( &finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost) ;
		
		}
		while(finished);
    
    
    		// Copy result from Device to host
   		cudaMemcpy( temp, d_cost, sizeof(int)*no_of_nodes, cudaMemcpyDeviceToHost) ;

		// Store results
		SSSP[0] = source;			
		for(int p=1;p<=no_of_nodes;p++)
			SSSP[p] = temp[p-1];	

		// Stop the timer
    		cudaEventRecord(end, 0);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&time, begin, end);
	
		// Record the time for each repetition
		total_time+=time;

		cudaEventDestroy(begin);
		cudaEventDestroy(end);
	
	
		// Store the result into a file
		//FILE *fpo = fopen("/home/mohsen/result-alg35.txt","w");
		//for(int j=1;j<=no_of_nodes;j++)
		//	fprintf(fpo, "%d  %d  %d\n", SSSP[0], j-1, SSSP[j]);
		//fclose(fpo);
		//printf("Results stored in result of SSSP(Harish).txt\n");
	
	
    		// Cleanup memory
   		free( h_graph_nodes);
   		free( h_graph_edges);
   		free( h_graph_mask1);
    		free( h_graph_weights);
		free( h_graph_mask2);
    		free( h_cost);
		free(temp);
		free(SSSP);
    		cudaFree(d_graph_nodes);
    		cudaFree(d_graph_edges);
    		cudaFree(d_graph_mask1);
    		cudaFree(d_graph_weights);
		cudaFree(d_graph_mask2);
    		cudaFree(d_cost);
		cudaFree(d_finished);	

	}

	// Take an average over total time
	printf( "Processing time: %f (ms)\n", (float)(total_time/repetition));

	return 0;
       
}

