
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <cstring>
#include <time.h>
#include <stdlib.h>

#include <omp.h>

#define Randmod(x) rand()%x
using namespace std;


struct pts_idx{
	int idx;
	int block_id;
	float inner;
};
bool increase_pts_idx(const pts_idx &first, const pts_idx &second){
	return first.block_id < second.block_id;
}

float is_in_box(float x, float y, float x_center, float y_center, float radius){
	if( fabs(x-x_center)<=radius && fabs(y-y_center)<=radius)
		return 1.0;
	else
		return 0.0;
}

struct num_idx{
	int num;
	int start_idx;
};


void randomsamlping(vector<pts_idx> _pts_idx, vector<num_idx> _num_idx, int MAX_NUM, int *sample_idx){
	int block_num = _num_idx.size();
	//cout<<sample_idx[MAX_NUM*block_num]<<endl;
	//cout<<"block_num: "<<block_num<<endl;
	#pragma omp parallel for
	for(int i=0;i<block_num; i++){
		srand((unsigned)time(NULL));
		for (int j=0;j<MAX_NUM;j++){			
			int tmp_id = _num_idx[i].start_idx + Randmod(_num_idx[i].num);
			sample_idx[i*MAX_NUM + j] = _pts_idx[tmp_id].idx;
		}
	}
}

extern "C" void fps_multiblocks(int totalnum_pts, int num_sample, int num_block, const int *num_pts_host, const int *start_id_host, const int *sort_id_host, const float *dataset_host, int *idxs_host);

void fps_multiblocks_warpper(vector<pts_idx> _pts_idx, vector<num_idx> _num_idx, float* xyz,int MAX_NUM, int *sample_idx){
	int totalnum_pts = _pts_idx.size();
	int num_block = _num_idx.size();
	int *num_pts = new int[num_block];
	int *start_id = new int[num_block];
	#pragma omp parallel for
	for(int i=0;i<num_block;i++){
		num_pts[i] = _num_idx[i].num;
		start_id[i] = _num_idx[i].start_idx;
	} 

	int *sort_id = new int[totalnum_pts];
	float *dataset = new float[totalnum_pts*3]; 
	#pragma omp parallel for
	for(int i=0;i<totalnum_pts;i++){
		int temp_id = _pts_idx[i].idx;
		sort_id[i] = temp_id;
		dataset[i*3] = xyz[temp_id*3];
		dataset[i*3+1] = xyz[temp_id*3+1];
		dataset[i*3+2] = xyz[temp_id*3+2];
	}

	fps_multiblocks(totalnum_pts, MAX_NUM, num_block, num_pts, start_id, sort_id, dataset, sample_idx);

	delete[] num_pts;
	delete[] start_id;
	delete[] sort_id;
	delete[] dataset;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C" void scene2blocks(float *pts_xyz, int *sample_idx, int &MAX_NUM, int &pts_num, int &x_num, float &block_size, int &count_threshold, int &block_num){

	//compute the idx of blocks
	vector<pts_idx> _pts_idx;
	pts_idx tmp_pts_idx;
	for (int i=0;i<pts_num; i++){
		tmp_pts_idx.idx = i;
		tmp_pts_idx.block_id = int( pts_xyz[i*3] / block_size) + int( pts_xyz[i*3+1] / block_size) * x_num;
		_pts_idx.push_back(tmp_pts_idx);
	}

	sort(_pts_idx.begin(), _pts_idx.end(), increase_pts_idx);

	//cout<<"pts_num: "<<pts_num<<endl;
	//cout<<"size: "<<_pts_idx.size()<<endl;
	//calculate the nummber of points in each block and their start id
	vector<num_idx> _num_idx;
	num_idx tmp_num_idx;
	int tmp_num=1;
	tmp_num_idx.start_idx = 0;
	for(int i=0;i<pts_num-1;i++){
		if(_pts_idx[i].block_id < _pts_idx[i+1].block_id){
			tmp_num_idx.num = tmp_num;
			_num_idx.push_back(tmp_num_idx);

			tmp_num_idx.start_idx = i+1;
			tmp_num = 0;
		}
		tmp_num += 1;
	}
	tmp_num_idx.num = tmp_num;
	_num_idx.push_back(tmp_num_idx);

    //cout<<"stastic done"<<endl;
	//erase blocks which has pts_xyz less than count_threshold
	vector<num_idx>::iterator it;
	for (it=_num_idx.begin();it!=_num_idx.end();){
		if(it->num < count_threshold)
			it = _num_idx.erase(it);
		else
			++it;
	}
	block_num = _num_idx.size();
	//
	fps_multiblocks_warpper( _pts_idx, _num_idx, pts_xyz, MAX_NUM, sample_idx);

	//randomsamlping(_pts_idx, _num_idx,  MAX_NUM,  sample_idx);
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C" void scene2blocks_withinner(float *pts_xyz, int *sample_idx, float *inner, int &MAX_NUM, int &pts_num, int &x_num, float &block_size, float &inner_radius, int &count_threshold, int &block_num){

	vector<pts_idx> _pts_idx;
	pts_idx tmp_pts_idx;
	int id_x, id_y;
	float x_center, y_center;
	for (int i=0;i<pts_num; i++){
		tmp_pts_idx.idx = i;
		id_x = int( pts_xyz[i*3] / block_size);
		id_y = int( pts_xyz[i*3+1] / block_size);
		tmp_pts_idx.block_id =  id_x + id_y * x_num;
		x_center = id_x*block_size + block_size/2;
		y_center = id_y*block_size + block_size/2;
		tmp_pts_idx.inner = is_in_box(pts_xyz[i*3], pts_xyz[i*3+1], x_center, y_center, inner_radius);
		_pts_idx.push_back(tmp_pts_idx);
	}

	vector<pts_idx> _pts_idx_copy(_pts_idx);
	sort(_pts_idx.begin(), _pts_idx.end(), increase_pts_idx);

	//calculate the nummber of points in each block and their start id
	vector<num_idx> _num_idx;
	num_idx tmp_num_idx;
	int tmp_num=0;
	tmp_num_idx.start_idx = 0;
	for(int i=0;i<pts_num-1;i++){
		if(_pts_idx[i].block_id < _pts_idx[i+1].block_id){
			tmp_num_idx.num = tmp_num;
			_num_idx.push_back(tmp_num_idx);

			tmp_num_idx.start_idx = i+1;
			tmp_num = 0;
		}
		tmp_num += 1;
	}
	tmp_num_idx.num = tmp_num;
	_num_idx.push_back(tmp_num_idx);

    //cout<<"stastic done"<<endl;
	//erase blocks which has pts_xyz less than count_threshold
	vector<num_idx>::iterator it;
	for (it=_num_idx.begin();it!=_num_idx.end();){
		if(it->num < count_threshold)
			it = _num_idx.erase(it);
		else
			++it;
	}
	
	//
	block_num = _num_idx.size();
	//cout<<sample_idx[MAX_NUM*block_num]<<endl;
	//cout<<"block_num: "<<block_num<<endl;
	fps_multiblocks_warpper( _pts_idx, _num_idx, pts_xyz, MAX_NUM, sample_idx);
	#pragma omp parallel for
	for(int i=0;i<block_num*MAX_NUM; i++){
		int tmp_id = sample_idx[i];
		inner[i] = _pts_idx_copy[tmp_id].inner;
	}	

	/*
	//random sampling
	#pragma omp parallel for
	for(int i=0;i<block_num; i++){
		srand((unsigned)time(NULL));
		for (int j=0;j<MAX_NUM;j++){			
			int tmp_id = _num_idx[i].start_idx + Randmod(_num_idx[i].num);
			sample_idx[i*MAX_NUM + j] = _pts_idx[tmp_id].idx;
			inner[i*MAX_NUM + j] = _pts_idx[tmp_id].inner;
		}
	}*/
	
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" void farthestpointsampling(int n, int m, const int random_init, float *temp_host, const float *dataset_host, int *idxs_host);
extern "C" void FPS_pythonwarpper(int &pts_num, int &sample_num, float *xyz, int *sample_idx){
	float *temp_host = new float[pts_num]; 
	memset(temp_host, 0, sizeof(float)*pts_num);

	int random_init = Randmod(pts_num);
	/*
	int *random_init = new int[b];
	srand((unsigned)time(NULL));
	for (int i=0;i<b;i++){
		random_init[i] = Randmod(pts_num);
	} */

	farthestpointsampling(pts_num, sample_num, random_init, temp_host, xyz, sample_idx);
}