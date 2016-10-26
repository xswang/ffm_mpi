#pragma once

#include <iostream>
#include <fstream>
#include <string.h>
#include <math.h>
#include <set>
#include "mpi.h"
#include <omp.h>

namespace dml{
typedef struct{
    float clk;
    float nclk;
    long idx;
} clkinfo;

class Predict{
    public:
    std::vector<std::set<int> > cross_field;
    Predict(LoadAllData* load_data, Param *param, int total_num_proc, int my_rank) 
            : data(load_data), param(param), nproc(total_num_proc), rank(my_rank){
        pctr = 0.0;
        MAX_ARRAY_SIZE = 1e6;
        g_all_non_clk = new float[MAX_ARRAY_SIZE];
        g_all_clk = new float[MAX_ARRAY_SIZE];
        g_nclk = new float[MAX_ARRAY_SIZE];
        g_clk = new float[MAX_ARRAY_SIZE];

        for(int i = 0; i < param->group; ++i){
            std::set<int> s;
            for(int j = 0; j < param->group; j += 1){
                s.insert(j);
            }
            cross_field.push_back(s);
        }
    }

    ~Predict(){
        delete[] g_all_non_clk;
        delete[] g_all_clk;
        delete[] g_nclk;
        delete[] g_clk;
    }

    double getElem(double* arr, int i, int j, int k){
        if(param->isfm)
            return arr[i * param->fea_dim + j + k];
        else return arr[i * param->fea_dim*param->group + j * param->group + k];
    }
    void print1dim(double* arr){
        for(int i = 0; i < param->factor * param->fea_dim * param->group; ++i)
        std::cout<<arr[i]<<std::endl;
    }
    void predict(double* glo_w, double* glo_v){
        int group = 0, index = 0; float value = 0.0; float pctr = 0.0;
        std::cout<<"test data size = "<<data->fea_matrix.size()<<std::endl;
        for(int i = 0; i < data->fea_matrix.size(); ++i) {
	        float wx = 0.0;
            for(int j = 0; j < data->fea_matrix[i].size(); ++j) {
                index = data->fea_matrix[i][j].fid;
                value = data->fea_matrix[i][j].val;
                wx += glo_w[index] * value;
            }
            std::set<int>::iterator setIter;
            for(int k = 0; k < param->factor; ++k){
                if(param->islr) break;
                float vxvx = 0.0, vvxx = 0.0;
                for(int col = 0; col < data->fea_matrix[i].size(); ++col){
                    group = data->fea_matrix[i][col].fgid;
                    index = data->fea_matrix[i][col].fid;
                    value = data->fea_matrix[i][col].val;
                    for(int f = 0; f < param->group; ++f){
                        setIter = cross_field[group].find(f);
                        if(setIter == cross_field[group].end()) continue;
                        if(param->isfm) f = 0;
                        double glov = getElem(glo_v, k, index, f);
                        vxvx += glov * value;
                        vvxx += glov * glov * value * value;
                        if(param->isfm) break;
                    }
                }
                vxvx *= vxvx;
                vxvx -= vvxx;
                wx += vxvx * 1.0 / 2.0;
            }
            //std::cout<<"wx = "<<wx<<std::endl;
            if(wx < -30){
                pctr = 1e-6;
            }
            else if(wx > 30){
                pctr = 1.0;
            }
            else{
                double ex = pow(2.718281828, wx);
                pctr = ex / (1.0 + ex);
            }
            int id = int(pctr*MAX_ARRAY_SIZE);
            clkinfo clickinfo;
            clickinfo.clk = data->label[i];
            clickinfo.nclk = 1 - data->label[i];
            clickinfo.idx = id;
            result_list.push_back(clickinfo);
        }
    }

    void merge_clk(){//merge local node`s clk
        memset(g_nclk, 0.0, MAX_ARRAY_SIZE * sizeof(float));
        memset(g_clk, 0.0, MAX_ARRAY_SIZE * sizeof(float));
        int cnt = result_list.size();
        for(int i = 0; i < cnt; ++i){
            long index = result_list[i].idx;
            g_nclk[index] += result_list[i].nclk;
            g_clk[index] += result_list[i].clk;
        }
    }

    int auc_cal(float* all_clk, float* all_nclk, double& auc_res){
            double clk_sum = 0.0;
            double nclk_sum = 0.0;
            double old_clk_sum = 0.0;
            double clksum_multi_nclksum = 0.0;
            auc_res = 0.0;
            for(int i = 0; i < MAX_ARRAY_SIZE; ++i){
                    old_clk_sum = clk_sum;
                    clk_sum += all_clk[i];
                    nclk_sum += all_nclk[i];
                    auc += (old_clk_sum + clk_sum) * all_nclk[i] / 2;
            }
            clksum_multi_nclksum = clk_sum * nclk_sum;
            auc_res = auc/(clksum_multi_nclksum);
    }

    int mpi_auc(int nprocs, int rank, double& auc){
        MPI_Status status;
        if(rank != 0){
            MPI_Send(g_nclk, MAX_ARRAY_SIZE, MPI_FLOAT, 0, 199, MPI_COMM_WORLD);
            MPI_Send(g_clk, MAX_ARRAY_SIZE, MPI_FLOAT, 0, 1999, MPI_COMM_WORLD);
        }
        else if(rank == 0){
            for(int i = 0; i < MAX_ARRAY_SIZE; ++i){
                g_all_non_clk[i] = g_nclk[i];
                g_all_clk[i] = g_clk[i];
            }
            for(int i = 1; i < nprocs; ++i){
                MPI_Recv(g_nclk, MAX_ARRAY_SIZE, MPI_FLOAT, i, 199, MPI_COMM_WORLD, &status);
                MPI_Recv(g_clk, MAX_ARRAY_SIZE, MPI_FLOAT, i, 1999, MPI_COMM_WORLD, &status);
                for(int i = 0; i < MAX_ARRAY_SIZE; ++i){
                    g_all_non_clk[i] += g_nclk[i];
                    g_all_clk[i] += g_clk[i];
                }
            }
            auc_cal(g_all_non_clk, g_all_clk, auc);
        }
    }

    void run(double* w, double* v){
        predict(w, v);
        merge_clk();
        mpi_auc(nproc, rank, auc);

        if(0 == rank){
            printf("AUC = %lf\n", auc);
        }
    }

    private:
    LoadAllData* data;
    Param *param;
    std::vector<clkinfo> result_list;
    int MAX_ARRAY_SIZE;
    double auc = 0.0;
    float* g_all_non_clk;
    float* g_all_clk;
    float* g_nclk;
    float* g_clk;
    float g_total_clk;
    float g_total_nclk;

    float pctr;
    int nproc; // total num of process in MPI comm world
    int rank; // my process rank in MPT comm world
};
}
