#pragma once
#include "learner.h"
#include "../predict.h"
#include <cblas.h>
#include <thread>
#include <mutex>
#include <time.h>
#include <omp.h>
#include "../threadpool/thread_pool.h"

namespace dml{
struct ThreadParam {
    int batchsize4thread;
};

class FtrlLearner : public Learner{
    public:
        FtrlLearner(LoadAllData *train_data, Predict* predict, Param *param, int nproc, int rank) 
                : Learner(param), data(train_data), pred(predict), param(param), nproc(nproc), rank(rank){
               Init(); 
        }
        ~FtrlLearner(){}

        void Init(){
            for(int i = 0; i < param->group; ++i){
                std::set<int> s;
                for(int j = 0; j < param->group; j += 1){
                        s.insert(j);
                }
                cross_field.push_back(s);
            }

            loc_g_tmp = new double[param->fea_dim];
            loc_g_v_tmp = new double[v_dim];

            alpha_v = 0.0;
            beta_v = 0.00001;
            lambda1_v = 0.000001;
            lambda2_v = 0.0;

            alpha = param->alpha;
            beta = param->beta;
            lambda1 = param->lambda1;
            lambda2 = param->lambda2;
        }

        void dump(int epoch){
            char buffer[1024];
            snprintf(buffer, 1024, "%d", epoch);
            std::string filename = buffer;
            std::ofstream md;
            md.open("./model/model_epoch" + filename + ".txt");
            if(!md.is_open()){
                    std::cout<<"save model open file error: "<< std::endl;
            }
            float wi;
            for(int j = 0; j < param->fea_dim; ++j){
                wi = loc_w[j];
                md<< j << "\t" <<wi<<std::endl;
            }
            md.close();
        }

        void batch_gradient_calculate();
        void batch_gradient_calculate_multithread(int start, int end);
        void allreduce_gradient();
        void allreduce_weight();

        void update_w(){
            #pragma omp parallel for
            for(int col = 0; col < param->fea_dim; ++col){
                loc_sigma[col] = ( sqrt (loc_n[col] + glo_g[col] * glo_g[col]) - sqrt(loc_n[col]) ) / param->alpha;
                loc_n[col] += glo_g[col] * glo_g[col];
                loc_z[col] += glo_g[col] - loc_sigma[col] * loc_w[col];
                if(abs(loc_z[col]) <= param->lambda1){
                    loc_w[col] = 0.0;
                }
                else{
                    float tmpr= 0.0;
                    if(loc_z[col] >= 0) tmpr = loc_z[col] - param->lambda1;
                    else tmpr = loc_z[col] + param->lambda1;
                    float tmpl = -1 * ( ( param->beta + sqrt(loc_n[col]) ) / param->alpha  + param->lambda2);
                    loc_w[col] = tmpr / tmpl;
                }
            }//end for
        }

        void update_v(){
            for(int k = 0; k < param->factor; ++k){
                if(param->islr) break;
                #pragma omp parallel for
                for(int col = 0; col < param->fea_dim; ++col){
                    for(int f = 0; f < param->group; ++f){
                        if(param->isfm) f = 0;
                        float old_locnv = getElem(loc_n_v, k, col, f);
                        float glogv = getElem(glo_g_v, k, col, f);
                        float locsigmav = (sqrt(old_locnv + glogv*glogv) - sqrt(old_locnv)) / alpha_v;

                        double new_locnv = old_locnv + glogv * glogv;
                        putVal(loc_n_v, new_locnv, k, col, f);
                        double old_loczv = getElem(loc_z_v, k, col, f);
                        double new_loczv = old_loczv + glogv - locsigmav * getElem(loc_v, k, col, f);
                        putVal(loc_z_v, new_loczv, k, col, f);
                        if(abs(new_loczv) <= lambda1_v){
                            putVal(loc_v, 0.0, k, col, f);
                        }
                        else{
                            float tmpr= 0.0;
                            if(new_loczv >= 0) tmpr = new_loczv - lambda1_v;
                            else tmpr = new_loczv + lambda1_v;
                            float tmpl = -1 * ( ( beta_v + sqrt(getElem(loc_n_v, k, col, f)) ) / alpha_v  + lambda2_v);
                            putVal(loc_v, tmpr / tmpl, k, col, f);
                        }
                        if(param->isfm) break;
                    }
                }//end for
            }//end for
        }//end update_v

        void run();

    public:
        std::mutex mutex;
        int row;
        int loc_g_nonzero;
        int loc_g_v_nonzero;
        double *loc_g_tmp;
        double *loc_g_v_tmp;
        float bias;

        float alpha;
        float beta;
        float lambda1;
        float lambda2;

        float alpha_v;
        float beta_v;
        float lambda1_v;
        float lambda2_v;

    public:
        MPI_Status status;
        LoadAllData *data;
        Predict *pred;
        Param *param;

        int nproc;
        int rank;
};
}
