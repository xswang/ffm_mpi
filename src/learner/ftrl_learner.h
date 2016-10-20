#pragma once
#include "learner.h"
#include "../predict.h"
#include <cblas.h>
#include <thread>
#include <mutex>

namespace DML{
class FTRL_learner : public Learner{
    public:
        FTRL_learner(LOAD_ALL_DATA *train_data, Predict* predict, Param *param, int nproc, int rank) 
                : Learner(param), data(train_data), pred(predict), param(param), nproc(nproc), rank(rank){
               Init(); 
        }

        ~FTRL_learner(){}

        void Init(){
            for(int i = 0; i < param->group; i++){
                std::set<int> s;
                for(int j = 0; j < param->group; j += 1){
                        s.insert(j);
                }
                cross_field.push_back(s);
            }

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
            for(int j = 0; j < param->fea_dim; j++){
                wi = loc_w[j];
                md<< j << "\t" <<wi<<std::endl;
            }
            md.close();
        }
        void batch_gradient_calculate_multithread(int thread_batch_size){
            std::cout<<"my rank = "<<rank<<std::endl;
                if(rank == 0)std::cout<<"hhhhhhhhh"<<std::endl;
            int group = 0, index = 0; float value = 0.0, pctr = 0.0;
            for(int line = 0; line < thread_batch_size; line++){
                if(rank == 0)std::cout<<" line ==== "<<line<<std::endl;
                float wx = bias;
                int ins_seg_num = data->fea_matrix[row].size();
                std::vector<float> vx_sum(param->factor, 0.0);
                float vxvx = 0.0, vvxx = 0.0;
                std::set<int>::iterator setIter;
                for(int col = 0; col < ins_seg_num; col++){//for one instance
                    group = data->fea_matrix[row][col].fgid;
                    index = data->fea_matrix[row][col].fid;
                    value = data->fea_matrix[row][col].val;
                    wx += loc_w[index] * value;
                    for(int k = 0; k < param->factor; k++){
                        if(param->islr) break;
                        for(int f = 0; f < param->group; f++){
                            setIter = cross_field[group].find(f);
                            if(setIter == cross_field[group].end()) continue;
                            if(param->isfm) f = 0;
                            int loc_v_temp = getElem(loc_v, k, index, f);
                            vx_sum[k] += loc_v_temp * value;
                            vvxx += loc_v_temp * loc_v_temp * value * value;
                            if(param->isfm) break;
                        }
                    }
                }//end for
                for(int k = 0; k < param->factor; k++){
                    if(param->islr) break;
                    vxvx += vx_sum[k] * vx_sum[k];
                }
                vxvx -= vvxx;
                wx += vxvx * 1.0 / 2.0;
                pctr = sigmoid(wx);
                float delta = pctr - data->label[row];
                for(int col = 0; col < ins_seg_num; col++){
                    group = data->fea_matrix[row][col].fgid;
                    index = data->fea_matrix[row][col].fid;
                    value = data->fea_matrix[row][col].val;
                    mutex.lock();
                    loc_g[index] += delta * value;
                    mutex.unlock();
                    float vx = 0.0;
                    for(int k = 0; k < param->factor; k++){
                        if(param->islr) break;
                        for(int f = 0; f < param->group; f++){
                            setIter = cross_field[group].find(f);
                            if(setIter == cross_field[group].end()) continue;
                            if(param->isfm) f = 0;
                            float tmpv = getElem(loc_v, k, index, f);
                            vx = tmpv * value;
                            mutex.lock();
                            addVal(loc_g_v, -1 * delta * (vx_sum[k] - vx) * value, k, index, f);
                            mutex.unlock();
                            if(param->isfm) break;
                        }
                    }
                }
                mutex.lock();
                row++;
                if(rank == 0)std::cout<<"row ==== "<<row<<std::endl;
                mutex.unlock();
            }//end for
        }//end batch_gradient_calculate_multithread

        void batch_gradient_calculate(){
                int group = 0, index = 0; float value = 0.0, pctr = 0.0;
                memset(loc_g, 0.0, param->fea_dim);//notation:
                memset(loc_g_v, 0.0, v_dim);//notation: 
                for(int line = 0; line < param->batch_size; line++){
                        float wx = bias;
                        int ins_seg_num = data->fea_matrix[row].size();
                        std::vector<float> vx_sum(param->factor, 0.0);
                        float vxvx = 0.0, vvxx = 0.0;
                        std::set<int>::iterator setIter;
                        for(int col = 0; col < ins_seg_num; col++){//for one instance
                                group = data->fea_matrix[row][col].fgid;
                                index = data->fea_matrix[row][col].fid;
                                value = data->fea_matrix[row][col].val;
                                wx += loc_w[index] * value;
                                for(int k = 0; k < param->factor; k++){
                                        if(param->islr) break;
                                        for(int f = 0; f < param->group; f++){
                                                setIter = cross_field[group].find(f);
                                                if(setIter == cross_field[group].end()) continue;
                                                if(param->isfm) f = 0;
                                                int loc_v_temp = getElem(loc_v, k, index, f);
                                                vx_sum[k] += loc_v_temp * value;
                                                vvxx += loc_v_temp * loc_v_temp * value * value;
                                                if(param->isfm) break;
                                        }
                                }
                        }//end for
                        for(int k = 0; k < param->factor; k++){
                                if(param->islr) break;
                                vxvx += vx_sum[k] * vx_sum[k]; 
                        }
                        vxvx -= vvxx;
                        wx += vxvx * 1.0 / 2.0;
                pctr = sigmoid(wx);
                float delta = pctr - data->label[row];
                for(int col = 0; col < ins_seg_num; col++){
                    group = data->fea_matrix[row][col].fgid;
                    index = data->fea_matrix[row][col].fid;
                    value = data->fea_matrix[row][col].val;
                    loc_g[index] += delta * value;
                    float vx = 0.0;
                    for(int k = 0; k < param->factor; k++){
                        if(param->islr) break;
                        for(int f = 0; f < param->group; f++){
                            setIter = cross_field[group].find(f);
                            if(setIter == cross_field[group].end()) continue;
                            if(param->isfm) f = 0;
                            float tmpv = getElem(loc_v, k, index, f);
                            vx = tmpv * value;
                            addVal(loc_g_v, -1 * delta * (vx_sum[k] - vx) * value, k, index, f);
                            if(param->isfm) break;
                        }
                    }
                }
                row++;
            }//end for
        }//end batch_gradient_calculate

        void allreduce_gradient(){
            cblas_dscal(param->fea_dim, 1.0/param->batch_size, loc_g, 1);
            loc_g_nonzero = filter(loc_g, param->fea_dim);
            std::vector<Send_datatype> loc_g_vec;
            filter_nonzero(loc_g, param->fea_dim, loc_g_vec);

            std::vector<Send_datatype> loc_g_v_vec;
            if(!param->islr){
                cblas_dscal(v_dim, 1.0/param->batch_size, loc_g_v, 1);
                loc_g_v_nonzero = filter(loc_g_v, v_dim);
                filter_nonzero(loc_g_v, v_dim, loc_g_v_vec);
            }
            if(rank != 0){
                MPI_Send(&loc_g_vec[0], loc_g_nonzero, newType, 0, 99, MPI_COMM_WORLD);
                if(!param->islr){
                    MPI_Send(&loc_g_v_vec[0], loc_g_v_nonzero, newType, 0, 399, MPI_COMM_WORLD);
                }
            }else if(rank == 0){
                cblas_dcopy(param->fea_dim, loc_g, 1, glo_g, 1);
                for(int r = 1; r < nproc; r++){
                    std::vector<Send_datatype> recv_loc_g_vec;
                    recv_loc_g_vec.resize(param->fea_dim);
                    MPI_Recv(&recv_loc_g_vec[0], param->fea_dim, newType, r, 99, MPI_COMM_WORLD, &status);
                    int recv_loc_g_num;
                    MPI_Get_count(&status, newType, &recv_loc_g_num);
                    for(int i = 0; i < recv_loc_g_num; i++){
                        int k = recv_loc_g_vec[i].key;
                        int v = recv_loc_g_vec[i].val;
                        glo_g[k] += v;
                    }
                    if(!param->islr){
                        cblas_dcopy(v_dim, loc_g_v, 1, glo_g_v, 1);
                        std::vector<Send_datatype> recv_loc_g_v_vec;
                        recv_loc_g_v_vec.resize(v_dim);
                        MPI_Recv(&recv_loc_g_v_vec[0], v_dim, newType, r, 399, MPI_COMM_WORLD, &status);
                        int recv_loc_g_v_num;
                        MPI_Get_count(&status, newType, &recv_loc_g_v_num);
                        for(int i = 0; i < recv_loc_g_v_num; i++){
                            int k = recv_loc_g_v_vec[i].key;
                            int v = recv_loc_g_v_vec[i].val;
                            glo_g_v[k] += v;
                        }
                    }
                }

                cblas_dscal(param->fea_dim, 1.0/nproc, glo_g, 1);
                update_w();
                if(!param->islr){
                    cblas_dscal(v_dim, 1.0/nproc, glo_g_v, 1);
                    update_v();
                }
            }
        }

        void allreduce_weight(){
            if(rank == 0){
                int loc_w_nonzero = filter(loc_w, param->fea_dim);

                std::vector<Send_datatype> loc_w_vec;
                filter_nonzero(loc_w, param->fea_dim, loc_w_vec);

                std::vector<Send_datatype> loc_v_vec;
                int loc_v_nonzero;
                if(!param->islr){
                    loc_v_nonzero = filter(loc_v, v_dim);
                    filter_nonzero(loc_v, v_dim, loc_v_vec);
                }
                for(int r = 1; r < nproc; r++){
                    MPI_Send(&loc_w_vec[0], loc_w_nonzero, newType, r, 999, MPI_COMM_WORLD);
                    if(!param->islr){
                        MPI_Send(&loc_v_vec[0], loc_v_nonzero, newType, r, 3999, MPI_COMM_WORLD);
                    }
                }
            }else if(rank != 0){
                std::vector<Send_datatype> recv_loc_w_vec;
                recv_loc_w_vec.resize(param->fea_dim);
                MPI_Recv(&recv_loc_w_vec[0], param->fea_dim, newType, 0, 999, MPI_COMM_WORLD, &status);
                int recv_loc_w_num;
                MPI_Get_count(&status, newType, &recv_loc_w_num);
                memset(loc_w, 0.0, param->fea_dim * sizeof(double));
                for(int i = 0; i < recv_loc_w_num; i++){
                    int k = recv_loc_w_vec[i].key;
                    int v = recv_loc_w_vec[i].val;
                    loc_w[k] = v;
                }

                if(param->islr != 1){
                    std::vector<Send_datatype> recv_loc_v_vec;
                    recv_loc_v_vec.resize(v_dim);
                    MPI_Recv(&recv_loc_v_vec[0], v_dim, newType, 0, 3999, MPI_COMM_WORLD, &status);
                    int recv_loc_v_num;
                    MPI_Get_count(&status, newType, &recv_loc_v_num);
                    memset(loc_v, 0.0, v_dim * sizeof(double));
                    for(int i = 0; i < recv_loc_v_num; i++){
                        int k = recv_loc_v_vec[i].key;
                        int v = recv_loc_v_vec[i].val;
                        loc_v[k] = v;
                    }
                }
            }
        }

        void update_w(){
            for(int col = 0; col < param->fea_dim; col++){
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
            for(int k = 0; k < param->factor; k++){
                if(param->islr) break;
                for(int col = 0; col < param->fea_dim; col++){
                    for(int f = 0; f < param->group; f++){
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
        }

        void run(){
            int batch_num = data->fea_matrix.size() / param->batch_size, batch_num_min = 0;
            MPI_Allreduce(&batch_num, &batch_num_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            std::cout<<"total epochs = "<<param->epoch<<" batch_num_min = "<<batch_num_min<<std::endl;
            for(int epoch = 0; epoch < param->epoch; epoch++){
                row = 0;
                int batches = 0;
                std::cout<<"epoch "<<epoch<<" ";
                std::cout<<"rankkkkkk is ====================== "<<rank<<std::endl;
                pred->run(loc_w, loc_v);
                std::cout<<" is ====================== "<<rank<<std::endl;
                if(rank == 0 && (epoch+1) % 20 == 0) dump(epoch);
                while(row < data->fea_matrix.size()){
                    if( (batches == batch_num_min - 1) ) break;
                    memset(loc_g, 0.0, param->fea_dim);//notation:
                    memset(loc_g_v, 0.0, v_dim);//notation:
                    //batch_gradient_calculate();
                    int core_num = std::thread::hardware_concurrency();
                    std::vector<std::thread> threads;
                    int thread_batchsize = param->batch_size / core_num;
                    ///* 
                    for(int i = 0; i < core_num; i++){
                        threads.push_back(std::thread(&FTRL_learner::batch_gradient_calculate_multithread, this, thread_batchsize));
                    }
                    for(auto &t : threads){
                        t.join();
                    }
                    //*/
                    if(row % 200000 == 0) std::cout<<"row = "<<row<<std::endl;
                    allreduce_gradient();
                    allreduce_weight();
                    batches++;
                }
            }
        }

    public:
        std::mutex mutex;
        int row;
        int loc_g_nonzero;
        int loc_g_v_nonzero;
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
        LOAD_ALL_DATA *data;
        Predict *pred;
        Param *param;

        int nproc;
        int rank;
};
}
