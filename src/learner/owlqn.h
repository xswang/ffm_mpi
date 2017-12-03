#ifndef OWLQN_H_
#define OWLQN_H_
#include "mpi.h"
#include <iostream>
#include <algorithm>
#include <pthread.h>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <string>
#include <math.h>
#include <stdlib.h>
#include <deque>
#include "load_data.h"
#include "predict.h"
#include <glog/logging.h>

#define MASTERID 0
#define NUM 999

extern "C"{
#include <cblas.h>
}

class OWLQN{
    public:
    OWLQN(Load_Data* ld, Predict* predict, int total_num_proc, int my_rank)
        : data(ld), pred(predict), num_proc(total_num_proc), rank(my_rank) {
            init();
    }

    ~OWLQN(){
        delete[] glo_w;
        delete[] glo_new_w;

        delete[] loc_wx;

        delete[] loc_g;
        delete[] glo_g;
        delete[] glo_new_g;
        delete[] glo_sub_g;
        delete[] glo_q;

        for(int i = 0; i < m; i++){
            delete[] glo_s_list[i];
            delete[] glo_y_list[i];
        }
        delete[] glo_s_list;
        delete[] glo_y_list;

        delete[] glo_alpha_list;
        delete[] glo_ro_list;
    }

    void init(){
        bias = 0.0;
        c = 1.0;
        glo_w = new double[data->glo_fea_dim]();
        glo_new_w = new double[data->glo_fea_dim]();
        for(int i = 0; i < data->glo_fea_dim; i++) {
            glo_w[i] = 0.0;
        }

        data->loc_ins_num = data->fea_matrix.size();
        loc_wx = new double[data->loc_ins_num]();

        loc_g = new double[data->glo_fea_dim]();
        glo_g = new double[data->glo_fea_dim]();
        glo_new_g = new double[data->glo_fea_dim]();
        glo_sub_g = new double[data->glo_fea_dim]();
        glo_q = new double[data->glo_fea_dim]();

        m = 10;
        now_m = 1;
        glo_s_list = new double*[m];
        for(int i = 0; i < m; i++){
            glo_s_list[i] = new double[data->glo_fea_dim]();
            for(int j = 0; j < data->glo_fea_dim; j++){
                glo_s_list[i][j] = glo_w[j];
            }
        }
        glo_y_list = new double*[m];
        for(int i = 0; i < m; i++){
            glo_y_list[i] = new double[data->glo_fea_dim]();
            for(int j = 0; j < data->glo_fea_dim; j++){
                glo_y_list[i][j] = glo_g[j];
            }
        }
        glo_alpha_list = new double[data->glo_fea_dim]();
        glo_ro_list = new double[data->glo_fea_dim]();

        loc_loss = 0.0;
        glo_loss = 0.0;
        loc_new_loss = 0.0;
        glo_new_loss = 0.0;

        lambda = 0.0001;
        backoff = 0.9;
        flag_wolf = 1;
    }

    void calculate_wx(double *w){
        long int idx = 0;
        int val = 0;
        for(int i = 0; i < data->fea_matrix.size(); i++) {
            loc_wx[i] = bias;
            for(int j = 0; j < data->fea_matrix[i].size(); j++) {
                idx = data->fea_matrix[i][j].idx;
                val = data->fea_matrix[i][j].val;
                loc_wx[i] += w[idx] * val;
            }
        }
    }

    double sigmoid(double x){
        if(x < -30){
            return 1e-6;
        }
        else if(x > 30){
            return 1.0;
        }
        else{
            double ex = pow(2.718281828, x);
            return ex / (1.0 + ex);
        }
    }

    double calculate_loss(double *w){
        double f = 0.0, single_loss = 0.0, regular_loss = 0.0;
        memset(loc_wx, 0, sizeof(double) * data->fea_matrix.size());
        calculate_wx(w);
        for(int i = 0; i < data->fea_matrix.size(); i++){
            single_loss = data->label[i] * log(sigmoid(loc_wx[i])) +
                      (1 - data->label[i]) * log(1 - sigmoid(loc_wx[i]));
            f += single_loss;
        }
        for(int j = 0; j < data->glo_fea_dim; j++){
            regular_loss += abs(w[j]);
        }
        return -f / data->fea_matrix.size() + regular_loss;
    }

    void calculate_gradient(double* g, double *w){
        int value;
        int index, single_feature_num, instance_num = data->fea_matrix.size();
        memset(g, 0.0, data->glo_fea_dim * sizeof(double));
        memset(loc_wx, 0, sizeof(double) * data->fea_matrix.size());
        calculate_wx(w);
        for(int i = 0; i < instance_num; i++){
            single_feature_num = data->fea_matrix[i].size();
            double y_h = data->label[i] - sigmoid(loc_wx[i]);
            for(int j = 0; j < single_feature_num; j++){
                index = data->fea_matrix[i][j].idx;
                value = data->fea_matrix[i][j].val;
                g[index] += y_h * value;
            }
        }
        for(int index = 0; index < data->glo_fea_dim; index++){
            g[index] = g[index] / instance_num;
        }
    }//end calculate_gradient

    void calculate_subgradient(){
        if(c == 0.0){
            for(int j = 0; j < data->glo_fea_dim; j++){
                glo_sub_g[j] = glo_g[j];
            }
        } else if(c != 0.0){
            for(int j = 0; j < data->glo_fea_dim; j++){
                if(glo_w[j] > 0){
                    glo_sub_g[j] = glo_g[j] + c;
                }
                else if(glo_w[j] < 0){
                    glo_sub_g[j] = glo_g[j] - c;
                }
                else {
                    if(glo_g[j] + c < 0){
                        glo_sub_g[j] = glo_g[j] - c;//左导数
                    } else if(glo_g[j] - c > 0){
                        glo_sub_g[j] = glo_g[j] + c;
                    } else {
                        glo_sub_g[j] = 0.0;
                    }
                }
            }
        }
    }

    void fix_dir_glo_q(){
        for(int j = 0; j < data->glo_fea_dim; ++j){
            if(glo_q[j] * glo_sub_g[j] < 0){
                glo_q[j] = 0.0;
            }
        }
    }

    void fix_dir_glo_new_w(){
        for(int j = 0; j < data->glo_fea_dim; j++){
            if(glo_new_w[j] * glo_w[j] < 0) glo_new_w[j] = 0.0;
            else glo_new_w[j] = glo_new_w[j];
        }
    }   

    void line_search(){
        flag_wolf = 1;
        int lo = 0;
        lambda = 0.05;
        while(true){
            std::cout<<"line search loop = "<<lo++<<std::endl;
            if(rank == MASTERID){
                for(int j = 0; j < data->glo_fea_dim; j++){
                    glo_new_w[j] = glo_w[j] +  lambda * glo_q[j];//change + to -
                }
                fix_dir_glo_new_w();//new_w subject to w in linesearch
                for(int r = 1; r < num_proc; r++){
                    MPI_Send(glo_new_w, data->glo_fea_dim, MPI_DOUBLE, r, 999, MPI_COMM_WORLD);
                }
            } else if(rank != MASTERID){
                MPI_Recv(glo_new_w, data->glo_fea_dim, MPI_DOUBLE, 0, 999, MPI_COMM_WORLD, &status);
            }

            loc_new_loss = calculate_loss(glo_new_w);

            if(rank != MASTERID){
                MPI_Send(&loc_new_loss, 1, MPI_DOUBLE, 0, 9999, MPI_COMM_WORLD);
            } 
            else if(rank == MASTERID){
                glo_new_loss = loc_new_loss;
                for(int r = 1; r < num_proc; r++){
                    MPI_Recv(&loc_new_loss, 1, MPI_DOUBLE, r, 9999, MPI_COMM_WORLD, &status);
                    glo_new_loss += loc_new_loss;
                }
                double wolf_pf = cblas_ddot(data->glo_fea_dim, (double*)glo_q, 1, (double*)glo_sub_g, 1);
                std::cout<<wolf_pf<<std::endl;
                //cblas_daxpy(data->glo_fea_dim, -1, (double*)glo_new_w, 1, (double*)glo_w, 1);
                glo_new_loss += 0.0001 * lambda * wolf_pf;
                std::cout<<"glo_loss "<<glo_loss<<" glo_new_loss "<<glo_new_loss<<std::endl;
                if(glo_new_loss <= glo_loss + 1e-6){
                    std::cout<<"wolf flog = "<<flag_wolf<<std::endl;
                    flag_wolf = 0;
                    for(int r = 1; r < num_proc; r++){
                        MPI_Send(&flag_wolf, 1, MPI_INT, r, 111, MPI_COMM_WORLD);
                    }
                }
                else{
                    flag_wolf = 1;
                    for(int r = 1; r < num_proc; r++){
                        MPI_Send(&flag_wolf, 1, MPI_INT, r, 111, MPI_COMM_WORLD);
                    }
                }
                lambda *= backoff;
            }//end masterid
            if(rank != MASTERID){
                MPI_Recv(&flag_wolf, 1, MPI_INT, 0, 111, MPI_COMM_WORLD, &status);
            }
            if(rank == 2)std::cout<<"step = " << step<<" myrank = "<<rank<<" flag_wolf = "<<flag_wolf<<" in linesearch~"<<std::endl;
            if ( flag_wolf == 0 ) break;
        }//end while
    }

    void print(double a[]){
        double max = 0.0;
        for(int j = 0; j < data->glo_fea_dim; j++){
            if (a[j] < max) max = a[j];
            if(a[j] != 0.0)std::cout<<"print function: a["<<j<<"] = "<<a[j]<<std::endl;
        }
        std::cout<<"-----------------------------------------"<<max<<std::endl;
    }

    void two_loop(){
        cblas_dcopy(data->glo_fea_dim, glo_sub_g, 1, glo_q, 1);
        if(now_m > m) now_m = m;
        for(int loop = now_m-2; loop >= 0; --loop){
            glo_ro_list[loop] = cblas_ddot(data->glo_fea_dim, &(*glo_y_list)[loop], 1, &(*glo_s_list)[loop], 1);
            glo_alpha_list[loop] = (cblas_ddot( data->glo_fea_dim, &(*glo_s_list)[loop], 1, (double*)glo_q, 1 ) + 1.0) / (glo_ro_list[loop] + 1.0);
            cblas_daxpy(data->glo_fea_dim, -1 * glo_alpha_list[loop], &(*glo_y_list)[loop], 1, (double*)glo_q, 1);
        }//end for
        double ydots = cblas_ddot(data->glo_fea_dim, glo_s_list[now_m - 2], 1, glo_y_list[now_m - 2], 1);
        double gamma = (ydots + 1.0)/ (glo_ro_list[now_m - 2] + 1.0);

        cblas_dscal(data->glo_fea_dim, gamma, (double*)glo_q, 1);

        for(int loop = 0; loop <= now_m-2; ++loop){
            double beta = (cblas_ddot(data->glo_fea_dim, &(*glo_y_list)[loop], 1, (double*)glo_q, 1) + 1.0) / (glo_ro_list[loop] + 1.0);
            cblas_daxpy(data->glo_fea_dim, glo_alpha_list[loop] - beta, &(*glo_s_list)[loop], 1, (double*)glo_q, 1);
        }//end for
    }

    void update_state(){
        calculate_gradient(glo_new_g, glo_new_w);

        //print(glo_new_w);
        cblas_daxpy(data->glo_fea_dim, -1, (double*)glo_w, 1, (double*)glo_new_w, 1);
        cblas_dcopy(data->glo_fea_dim, (double*)glo_new_w, 1, (double*)glo_s_list[(now_m-1) % m], 1);
        cblas_daxpy(data->glo_fea_dim, 1, (double*)glo_w, 1, (double*)glo_new_w, 1);
        std::swap(glo_w, glo_new_w);

        cblas_daxpy(data->glo_fea_dim, -1, (double*)glo_g, 1, (double*)glo_new_g, 1);
        cblas_dcopy(data->glo_fea_dim, (double*)glo_new_g, 1, (double*)glo_y_list[(now_m-1) % m], 1);
        //double a = cblas_ddot(data->glo_fea_dim, &(*glo_y_list)[0], 1, &(*glo_s_list)[0], 1);
        //std::cout<<" a ==== "<<a<<std::endl;
        now_m++;
        //glo_loss = glo_new_loss;
    }

    bool meet_criterion(){
        if(step == 300) return true;
        return false;
    } 

    void save_model() {
        if(MASTERID == rank) {
            time_t rawtime;
            struct tm* timeinfo;
            char buffer[80];
            time(&rawtime);
            timeinfo = localtime(&rawtime);
            strftime(buffer, 80, "%Y%m%d_%H%M%S", timeinfo);
            std::string time_s = buffer;

            std::ofstream md;
            md.open("./model/lr_model_" + time_s + ".txt");
            double wi = 0.0;
            for(int i = 0; i < data->glo_fea_dim; ++i) {
                wi = glo_new_w[i];
                md << i << ':' << wi;
                if(i != data->glo_fea_dim - 1){
                    md << ' ';
                }
            }
            md.close();
        }
    }

    void owlqn(){
        for(step = 0; step < steps; step++){
            std::cout<<"step = "<<step<<" rank = "<<rank<<std::endl;
            loc_loss = calculate_loss(glo_w);
            calculate_gradient(loc_g, glo_w);
            if(rank != MASTERID){
                MPI_Send(&loc_loss, 1, MPI_DOUBLE, MASTERID, 90, MPI_COMM_WORLD);
                MPI_Send(loc_g, data->glo_fea_dim, MPI_DOUBLE, MASTERID, 99, MPI_COMM_WORLD);
            }
            else if(rank == MASTERID){
                glo_loss = loc_loss;
                for(int j = 0; j < data->glo_fea_dim; j++){
                    glo_g[j] = loc_g[j];
                }
                for(int r = 1; r < num_proc; r++){
                    MPI_Recv(&loc_loss, 1, MPI_DOUBLE, r, 90, MPI_COMM_WORLD, &status);
                    glo_loss += loc_loss;
                    MPI_Recv(loc_g, data->glo_fea_dim, MPI_DOUBLE, r, 99, MPI_COMM_WORLD, &status);
                    for(int j = 0; j < data->glo_fea_dim; j++){
                        glo_g[j] += loc_g[j];
                    }
                }
                for(int j = 0; j < data->glo_fea_dim; j++){
                    glo_g[j] /= num_proc;
                }
                calculate_subgradient();
                if(now_m != 1){
                    std::cout<<"now_m = "<<now_m<<std::endl;
                    two_loop();
                    fix_dir_glo_q();
                }
                else{
                    std::cout<<"now_m = "<<now_m<<std::endl;
                    cblas_dcopy(data->glo_fea_dim, glo_sub_g, 1, glo_q, 1);
                }
            }//end if
            line_search();
            /*
            for(int j = 0; j < data->glo_fea_dim; j++){
                glo_new_w[j] = glo_w[j] - lambda * glo_q[j];//change + to -
            }*/
            fix_dir_glo_new_w();
            if(rank == MASTERID){
                update_state();
            }//end if
            pred->run(glo_w);
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }//end owlqn

    double* glo_w; //global model parameter
    int steps;
    int step;
    int batch_size;
    private:
    MPI_Status status;
    Load_Data* data;
    Predict* pred;
    int flag_wolf;
    int num_proc; // total num of process in MPI comm world
    int rank; // my process rank in MPT comm world

    double bias;
    double c; //l1 norm parameter
    double* glo_new_w; //model paramter after line search
    double* loc_wx; //z = W*Xi, z is input for sigmoid(z)
    double* loc_g; //gradient of loss function compute by data on this process
    double* glo_g; //gradient of loss function compute by data on all process
    double* glo_new_g;

    double* glo_sub_g; //global sub gradient
    double* glo_q; //global search direction

    int m; //number memory data we want in owlqn(lbfgs)
    int now_m; //num of memory data we got now
    double** glo_s_list; //global s list in lbfgs two loop
    double** glo_y_list; //global y list in lbfgs two loop
    double* glo_alpha_list; //global alpha list in lbfgs two loop
    double* glo_ro_list; //global ro list in lbfgs two loop

    double loc_loss; //local loss
    double glo_loss; //global loss
    double loc_new_loss; //new local loss
    double glo_new_loss; //new global loss

    double lambda; //learn rate in line search
    double backoff; //back rate in line search
};
#endif
