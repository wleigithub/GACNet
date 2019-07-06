
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <string>
#include <math.h>
#include <time.h>
#include <stdlib.h>

#include <omp.h>
#include <flann/flann.h>

using namespace std;


void weight_cal(int n,  float *rgb, float *dist_spatial,  int *idx, float *weight) {
    #pragma omp parallel for
    for(int i=0; i<n; i++){
        float color_r = rgb[i*3];
        float color_g = rgb[i*3+1];
        float color_b = rgb[i*3+2];
        float dist_spt[3]={0},dist_col[3]={0}; 
        
        for (int j=0;j<3;j++){
            int tmp_id;
            tmp_id = idx[i*3+j];
            float near_r = rgb[tmp_id*3];
            float near_g = rgb[tmp_id*3+1];
            float near_b = rgb[tmp_id*3+2];
            
            dist_col[j] = 1.0 / max(sqrt( (near_r-color_r)*(near_r-color_r) + (near_g-color_g)*(near_g-color_g) + (near_b-color_b)*(near_b-color_b) ), float(1e-8));
            dist_spt[j] = 1.0 / max(dist_spatial[i*3 + j], float(1e-8));
            //printf("%f %f\t", dist_col[j], dist_spt[j]);
        }

        float norm_color = dist_col[0]+dist_col[1]+dist_col[2];
        float norm_spatial = dist_spt[0] + dist_spt[1] + dist_spt[2];

        for (int j=0;j<3;j++){
            dist_col[j] /=  norm_color;
            dist_spt[j] /= norm_spatial;
            weight[i*3 + j] = 0.7* dist_col[j] + 0.3*dist_spt[j];          
        }
    }
}


extern "C" void interpolateLauncher(int n_host, int m_host, int c_host, float *features_host, int *idx_host, float *weight_host, float *probs_host);
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C" void three_interpolate(float *pts, float *rgb, float *features, int *sample_ids, float *probs, int &point_num, int &sample_num, int &class_num){

    float *dataset = new float[sample_num*3];
    #pragma omp parallel for
    for(int i=0;i<sample_num;i++){
        int tmp_id = sample_ids[i];
        dataset[i*3] = pts[tmp_id*3];
        dataset[i*3+1] = pts[tmp_id*3+1];
        dataset[i*3+2] = pts[tmp_id*3+2];
    }

    int *knn_idx = new int[point_num*3];
    float *dist_spatial = new float[point_num*3];
    memset(knn_idx, 0, (point_num*3)*sizeof(int));
	memset(dist_spatial, 0, (point_num*3)*sizeof(float));

	struct FLANNParameters p;
	float speedup;
	flann_index_t index_id;
	p = DEFAULT_FLANN_PARAMETERS;
	p.algorithm = FLANN_INDEX_KDTREE;
	p.trees = 1;
	p.log_level = FLANN_LOG_INFO;
	p.checks = 1024;

    cout<<"searching..."<<endl;
	index_id = flann_build_index(dataset, sample_num, 3, &speedup, &p);
	flann_find_nearest_neighbors_index(index_id, pts, point_num, knn_idx, dist_spatial, 3, &p);

    //cout<<"point_num: "<< point_num<<endl;
    cout<<"weight calculating... "<<endl;
    float *weight = new float[point_num*3];
    memset(weight, 0, (point_num*3)*sizeof(float));
    weight_cal(point_num, rgb, dist_spatial, knn_idx, weight);

    cout<<"interpolate start..."<<endl;
    interpolateLauncher(point_num, sample_num, class_num, features, knn_idx, weight, probs);
    cout<<"interpolate done!"<<endl;

    delete[] dataset;
    delete[] knn_idx;
    delete[] dist_spatial;
    delete[] weight;
}



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void weight_cal_geo(int n, int k_num, float *geometry, float *dist_spatial,  int *idx, float *weight) {
    #pragma omp parallel for
    for(int i=0; i<n; i++){
        float geo[4] = {geometry[i*4], geometry[i*4+1], geometry[i*4+2], geometry[i*4+3]};
        float dist_spt[k_num]={0},dist_geo[k_num]={0}; 
        
        float norm_geo = 1e-8;
        float norm_spt = 1e-8;
        for (int j=0;j<k_num;j++){
            int tmp_id = idx[i*k_num+j];
            float diff_geo[4] = { geometry[tmp_id*4]-geo[0], geometry[tmp_id*4+1]-geo[1], geometry[tmp_id*4+2]-geo[2], geometry[tmp_id*4+3]-geo[3] };

            dist_geo[j] = 1.0 / max(sqrt( diff_geo[0]*diff_geo[0] + diff_geo[1]*diff_geo[1] + diff_geo[2]*diff_geo[2] + diff_geo[3]*diff_geo[3] ), float(1e-8));
            dist_spt[j] = 1.0 / max(dist_spatial[i*k_num + j], float(1e-8));
            //printf("%f %f\t", dist_col[j], dist_spt[j]);
            norm_geo += dist_geo[j];
            norm_spt += dist_spt[j];
        }

        for (int j=0;j<k_num;j++){
            dist_geo[j] /=  norm_geo;
            dist_spt[j] /= norm_spt;
            weight[i*k_num + j] = 0.5* dist_geo[j] + 0.5*dist_spt[j];          
        }
    }
}


extern "C" void filterLauncher(int n_host, int c_host, int k_num, int iter_num, float *features_host, int *idx_host, float *weight_host, float *probs_host);
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C" void geo_filter(float *pts, float *geometry, float *features, float *probs, int &point_num, int &class_num, int &k_num, int &iter_num){

    int *knn_idx = new int[point_num*k_num];
    float *dist_spatial = new float[point_num*k_num];
    memset(knn_idx, 0, (point_num*k_num)*sizeof(int));
	memset(dist_spatial, 0, (point_num*k_num)*sizeof(float));

	struct FLANNParameters p;
	float speedup;
	flann_index_t index_id;
	p = DEFAULT_FLANN_PARAMETERS;
	p.algorithm = FLANN_INDEX_KDTREE;
	p.trees = 4;
	p.log_level = FLANN_LOG_INFO;
	p.checks = 1000;

    cout<<"knn searching..."<<endl;
	index_id = flann_build_index(pts, point_num, 3, &speedup, &p);
	flann_find_nearest_neighbors_index(index_id, pts, point_num, knn_idx, dist_spatial, k_num, &p);

    //cout<<"point_num: "<< point_num<<endl;
    
    float *weight = new float[point_num*k_num];
    memset(weight, 0, (point_num*k_num)*sizeof(float));

    cout<<"weight calculating... "<< endl;
    weight_cal_geo(point_num, k_num, geometry, dist_spatial, knn_idx, weight);

    cout<<"filter start..."<<endl;
    filterLauncher(point_num, class_num, k_num, iter_num, features, knn_idx, weight, probs);
    cout<<"filter done!"<<endl;

    delete[] knn_idx;
    delete[] dist_spatial;
    delete[] weight;
}