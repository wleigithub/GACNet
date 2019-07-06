
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
#include <eigen3/Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <omp.h>
#include <map>  

using namespace std;

struct eigen_index{
	float eigen_value;
	int index;
};
bool my_decrease_sort(const eigen_index &first, const eigen_index &second){
	return first.eigen_value > second.eigen_value;
}

const int eig_num = 6;

////////////////////////////////////////////////////////////////////////////////////
void data2cloud(float *dataset, int num, pcl::PointCloud<pcl::PointXYZ>::Ptr Cloud){
	pcl::PointXYZ tmpPoint;
    for(int i=0;i<num;++i){
        tmpPoint.x = dataset[i*3];
        tmpPoint.y = dataset[i*3+1];
        tmpPoint.z = dataset[i*3+2];
        Cloud->points.push_back(tmpPoint);
    }
}

////////////////////////////////////////////////////////////////////////////////////
void compute_eiginf(std::vector<int> pointIdxRadiusSearch, pcl::PointCloud<pcl::PointXYZ>::Ptr Cloud, float *eigen_inf){
	//matrix and cloud load
    Eigen::MatrixXf observe = Eigen::MatrixXf::Random(pointIdxRadiusSearch.size(), 3);
	for (size_t ri = 0; ri < pointIdxRadiusSearch.size(); ++ri){
		pcl::PointXYZ tmp_point;
		observe(ri, 0) = Cloud->points[pointIdxRadiusSearch[ri]].x;
		observe(ri, 1) = Cloud->points[pointIdxRadiusSearch[ri]].y;
		observe(ri, 2) = Cloud->points[pointIdxRadiusSearch[ri]].z;
	}
	//caculate eigen value, eigen vector 
	Eigen::MatrixXf centered = observe.rowwise() - observe.colwise().mean();
	Eigen::Matrix3f covMat = (centered.adjoint() * centered) / double(observe.rows() - 1);
	Eigen::EigenSolver<Eigen::Matrix3f> es(covMat);
	Eigen::MatrixXf D = es.pseudoEigenvalueMatrix();
	Eigen::MatrixXf V = es.pseudoEigenvectors();
	//sort eigens from big to small
	std::vector<eigen_index> eigen_indexs;
	eigen_index tmp_eigen_index;
	for (int i = 0; i < 3; i++){
		tmp_eigen_index.eigen_value = D(i, i);
		tmp_eigen_index.index = i;
		eigen_indexs.push_back(tmp_eigen_index);
	}
	sort(eigen_indexs.begin(), eigen_indexs.end(), my_decrease_sort);

	//eigen normlization
	float sum_deno = max(sqrt(D(0, 0)*D(0, 0) + D(1, 1)*D(1, 1) + D(2, 2)*D(2, 2)),  FLT_MIN);
	//float nor_eig[3] = { 0 };
	//normlized eigen values
	eigen_inf[0] = max(fabs(D(eigen_indexs[0].index, eigen_indexs[0].index) / sum_deno), FLT_MIN);
	eigen_inf[1] = max(fabs(D(eigen_indexs[1].index, eigen_indexs[1].index) / sum_deno), FLT_MIN);
	eigen_inf[2] = max(fabs(D(eigen_indexs[2].index, eigen_indexs[2].index) / sum_deno), FLT_MIN);
	//norm vector
	eigen_inf[3] = fabs(V(0, eigen_indexs[2].index));
	eigen_inf[4] = fabs(V(1, eigen_indexs[2].index));
	eigen_inf[5] = fabs(V(2, eigen_indexs[2].index));
    //printf("%f", eigen_inf[0]);
}

////////////////////////////////////////////////////////////////////////////////////
extern "C" void compute_eigen(float *nearpoints, int &point_num, float *seachpoints, int &search_num, float &radius, int &k_num, float *eigen_inf){
	//matrix and cloud load

  	pcl::PointCloud<pcl::PointXYZ>::Ptr Cloud(new pcl::PointCloud<pcl::PointXYZ>);
    data2cloud(nearpoints, point_num, Cloud);

    //main program
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(Cloud);

    #pragma omp parallel for
    for (int i=0;i<search_num;++i){
        pcl::PointXYZ searchPoint;
        std::vector<int> pointIdxRadiusSearch(k_num);
        std::vector<float> pointRadiusSquaredDistance(k_num);
        float tmp[eig_num] ={0};

        searchPoint.x = Cloud->points[i].x;
        searchPoint.y = Cloud->points[i].y;
        searchPoint.z = Cloud->points[i].z;

		/*
        if (kdtree.nearestKSearch(searchPoint, k_num, pointIdxNKNSearch, pointNKNSquaredDistance)>3){//if the neighbors are less than 3, we cannot use them for norm calculation
            compute_eiginf(pointIdxNKNSearch, Cloud, tmp);
        }else{
            cout<<"KNN search failed!"<<endl;
        }
		*/
		
        if (kdtree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > k_num){
            compute_eiginf(pointIdxRadiusSearch, Cloud, tmp);
        }
        else{
            kdtree.nearestKSearch(searchPoint, k_num, pointIdxRadiusSearch, pointRadiusSquaredDistance);
            compute_eiginf(pointIdxRadiusSearch, Cloud, tmp);
        }
		
        for(int j=0;j<eig_num;++j)
            eigen_inf[i*eig_num+j] = tmp[j];
    }

}  

////////////////////////////////////////////////////////////////////////////////////
extern "C" void eiginf_mapping(float *eiginf, float *eiginf_batch, int *unique_idx, int &unique_num, int *idx, int &pts_num){
	map<int, int> mapeiginf;  
	//#pragma omp parallel for
	for(int i=0;i<unique_num;i++)
		mapeiginf[unique_idx[i]] = i;
	
	#pragma omp parallel for
	for(int i=0;i<pts_num;i++){
		int tmp_id = mapeiginf[idx[i]] * eig_num;
		for(int j=0;j<eig_num;j++)
			eiginf_batch[i*eig_num + j] = eiginf[tmp_id+j];
	}
}  

////////////////////////////////////////////////////////////////////////////////////
extern "C" void eig_cal_map(float *points, int &point_num, float *seachpoints, int &search_num, int *unique_idx, int *idx, int &sampled_num, float &radius, int &k_num, float *eiginf_batch){

	float *eigen_inf= new float[search_num *eig_num];
	compute_eigen(points, point_num, seachpoints, search_num, radius, k_num, eigen_inf);

	cout<<"feature mapping..."<<endl;
	eiginf_mapping(eigen_inf, eiginf_batch, unique_idx, search_num, idx, sampled_num);

	delete[] eigen_inf;
}  





////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////    next functions are for point cloud with intensity ///////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
const int eig_intensity_num = 4;
void compute_eiginf_intensity(std::vector<int> pointIdxRadiusSearch, pcl::PointCloud<pcl::PointXYZ>::Ptr Cloud, float *intensity, float *eigen_inf){
	//matrix and cloud load
    Eigen::MatrixXf observe = Eigen::MatrixXf::Random(pointIdxRadiusSearch.size(), 3);
	float *tmp_intensity = new float[pointIdxRadiusSearch.size()];
	float sum = 0;
	for (size_t ri = 0; ri < pointIdxRadiusSearch.size(); ++ri){
		pcl::PointXYZ tmp_point;
		observe(ri, 0) = Cloud->points[pointIdxRadiusSearch[ri]].x;
		observe(ri, 1) = Cloud->points[pointIdxRadiusSearch[ri]].y;
		observe(ri, 2) = Cloud->points[pointIdxRadiusSearch[ri]].z;
		tmp_intensity[ri] = intensity[pointIdxRadiusSearch[ri]];
		sum += tmp_intensity[ri];
	}
	float intensity_mean = sum/ (pointIdxRadiusSearch.size()+1e-8);
	sum = 0; 
	for (size_t i = 0; i < pointIdxRadiusSearch.size(); ++i){
		sum += (tmp_intensity[i]-intensity_mean) * (tmp_intensity[i]-intensity_mean) ;
	}
	float intensity_var = sum/ (pointIdxRadiusSearch.size()+1e-8);

	//caculate eigen value, eigen vector 
	Eigen::MatrixXf centered = observe.rowwise() - observe.colwise().mean();
	Eigen::Matrix3f covMat = (centered.adjoint() * centered) / double(observe.rows() - 1);
	Eigen::EigenSolver<Eigen::Matrix3f> es(covMat);
	Eigen::MatrixXf D = es.pseudoEigenvalueMatrix();
	Eigen::MatrixXf V = es.pseudoEigenvectors();
	//sort eigens from big to small
	std::vector<eigen_index> eigen_indexs;
	eigen_index tmp_eigen_index;
	for (int i = 0; i < 3; i++){
		tmp_eigen_index.eigen_value = D(i, i);
		tmp_eigen_index.index = i;
		eigen_indexs.push_back(tmp_eigen_index);
	}
	sort(eigen_indexs.begin(), eigen_indexs.end(), my_decrease_sort);

	//eigen normlization
	double sum_deno = sqrt(D(0, 0)*D(0, 0) + D(1, 1)*D(1, 1) + D(2, 2)*D(2, 2));
	eigen_inf[0] = fabs(D(eigen_indexs[0].index, eigen_indexs[0].index) / (sum_deno + 0.00000001));
	eigen_inf[1] = fabs(D(eigen_indexs[1].index, eigen_indexs[1].index) / (sum_deno + 0.00000001));
	eigen_inf[2] = fabs(D(eigen_indexs[2].index, eigen_indexs[2].index) / (sum_deno + 0.00000001));
	eigen_inf[3] = intensity_var;
}

////////////////////////////////////////////////////////////////////////////////////
extern "C" void compute_eigen_intensity(float *nearpoints, int &point_num, float *intensity, float *seachpoints, int &search_num, float &radius, int &k_num, float *eigen_inf){
	//matrix and cloud load

  	pcl::PointCloud<pcl::PointXYZ>::Ptr Cloud(new pcl::PointCloud<pcl::PointXYZ>);
    data2cloud(nearpoints, point_num, Cloud);

    //main program
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(Cloud);

    #pragma omp parallel for
    for (int i=0;i<search_num;++i){
        pcl::PointXYZ searchPoint;
        std::vector<int> pointIdxRadiusSearch;
        std::vector<float> pointRadiusSquaredDistance;
        float tmp[eig_intensity_num] ={0};

        searchPoint.x = seachpoints[i*3];
        searchPoint.y = seachpoints[i*3+1];
        searchPoint.z = seachpoints[i*3+2];

        if (kdtree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > k_num){
            compute_eiginf_intensity(pointIdxRadiusSearch, Cloud, intensity, tmp);
        }
        else{
            kdtree.nearestKSearch(searchPoint, k_num, pointIdxRadiusSearch, pointRadiusSquaredDistance);
            compute_eiginf_intensity(pointIdxRadiusSearch, Cloud, intensity, tmp);
        }

        for(int j=0;j<eig_intensity_num;++j)
            eigen_inf[i*eig_intensity_num+j] = tmp[j];
    }

}  

////////////////////////////////////////////////////////////////////////////////////
extern "C" void eiginf_mapping_intensity(float *eiginf, float *eiginf_batch, int *unique_idx, int &unique_num, int *idx, int &pts_num){
	map<int, int> mapeiginf;  
	//#pragma omp parallel for
	for(int i=0;i<unique_num;i++)
		mapeiginf[unique_idx[i]] = i;
	
	#pragma omp parallel for
	for(int i=0;i<pts_num;i++){
		int tmp_id = mapeiginf[idx[i]] * eig_intensity_num;
		for(int j=0;j<eig_intensity_num;j++)
			eiginf_batch[i*eig_intensity_num + j] = eiginf[tmp_id+j];
	}
}  

////////////////////////////////////////////////////////////////////////////////////
extern "C" void eig_cal_map_intensity(float *points, int &point_num, float *intensity, float *seachpoints, int &search_num, int *unique_idx, int *idx, int &sampled_num, float &radius, int &k_num, float *eiginf_batch){

	float *eigen_inf= new float[search_num *eig_intensity_num];
	compute_eigen_intensity(points, point_num, intensity, seachpoints, search_num, radius, k_num, eigen_inf);

	cout<<"feature mapping..."<<endl;
	eiginf_mapping_intensity(eigen_inf, eiginf_batch, unique_idx, search_num, idx, sampled_num);

	delete[] eigen_inf;
}  
