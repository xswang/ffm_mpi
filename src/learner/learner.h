#pragma once
#include <gflags/gflags.h>
#include "../param.h"
#include <math.h>
#include <vector>
#include <set>
#include "mpi.h"
#include <omp.h>

namespace dml{
class Send_datatype{
    public:
        double key;
        double val;
};

class Learner{
    public:
        Learner(Param *param) : param(param){
            if(param->islr) v_dim = 0;
            else if(param->isfm) v_dim = param->factor * param->fea_dim * 1;
            else if(param->isffm) v_dim = param->factor * param->fea_dim * param->group;
            loc_w = new double[param->fea_dim]();
            loc_g = new double[param->fea_dim]();
            glo_g = new double[param->fea_dim]();
            loc_sigma = new double[param->fea_dim]();
            loc_n = new double[param->fea_dim]();
            loc_z = new double[param->fea_dim]();

            loc_v = new double[v_dim]();
            for(int i = 0; i < v_dim; i++){
                loc_v[i] = gaussrand();
            }
            loc_g_v = new double[v_dim]();
            glo_g_v = new double[v_dim];
            loc_sigma_v = new double[v_dim]();
            loc_n_v = new double[v_dim]();
            loc_z_v = new double[v_dim]();

            int block_length[] = {1, 1};
            MPI::Datatype oldType[] = {MPI_DOUBLE, MPI_DOUBLE};
            MPI::Aint addressOffsets[] = {0, 1 * sizeof(double)};
            newType = MPI::Datatype::Create_struct(
                            sizeof(block_length) / sizeof(int),
                            block_length,
                            addressOffsets,
                            oldType
                            );
            newType.Commit();
        }
        ~Learner(){}
        virtual void Init() = 0;
        virtual void batch_gradient_calculate() = 0;
        virtual void update_w() = 0;
        virtual void update_v() = 0;
        virtual void dump(int epoch) = 0;
    public:
        double gaussrand(){
                static double V1, V2, S;
                static int phase = 0;
                double X;
                if ( phase == 0 ) {
                        do {
                    double U1 = (double)rand() / RAND_MAX;
                    double U2 = (double)rand() / RAND_MAX;
                    V1 = 2 * U1 - 1;
                    V2 = 2 * U2 - 1;
                    S = V1 * V1 + V2 * V2;
                } while(S >= 1 || S == 0);
                X = V1 * sqrt(-2 * log(S) / S);
            }
            else{
                X = V2 * sqrt(-2 * log(S) / S);
            }
            phase = 1 - phase;
            return X * 0.1 + 0.0;
        }

        float sigmoid(float x){
            if(x < -30) return 1e-6;
            else if(x > 30) return 1.0;
            else{
                double ex = pow(2.718281828, x);
                return ex / (1.0 + ex);
            }
        }

        double getElem(double* arr, int i, int j, int k){
            if(param->isfm) return arr[i * param->fea_dim + j + k];
            else return arr[i * param->fea_dim*param->group + j * param->group + k];
        }

        void putVal(double* arr, float val, int i, int j, int k){
                if(param->isfm) arr[i*param->fea_dim + j + k] = val;
                else arr[i*param->fea_dim*param->group + j * param->group + k] = val;
        }

        void addVal(double* arr, int val, int i, int j, int k){
                if(param->isfm) arr[i * param->fea_dim + j + k] += val;
                else arr[i * param->fea_dim*param->group + j * param->group + k] += val;
        }

        long int filter(double* a, long int n){
            int nonzero = 0;
            //#pragma omp parallel for   
            for(int i = 0; i < n; ++i){
                if(a[i] != 0.0) nonzero += 1;
            }
            return nonzero;
        }
        void filter_nonzero(double *a, long int n, std::vector<Send_datatype> &vec){
            Send_datatype dt;
            //#pragma omp parallel for
            for(int i = 0; i < n; ++i){
                if(a[i] != 0.0){
                    dt.key = i;
                    dt.val = a[i];
                    vec.push_back(dt);
                }
            }
        }

    public:
        MPI::Datatype newType;
        Param *param;
        std::vector<std::set<int> > cross_field;

        int v_dim;

        double *loc_w;
        double *loc_v;
        double* loc_g;
        double* glo_g;
        double* loc_z;
        double* loc_sigma;
        double* loc_n;

        double* loc_g_v;
        double* glo_g_v;
        double* loc_sigma_v;
        double* loc_n_v;
        double* loc_z_v;
};
}
