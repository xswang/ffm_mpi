#ifndef FTRL_H
#define FTRL_H
#include "load_data.h"
#include "predict.h"
#include "mpi.h"
#include <math.h>
#include <cblas.h>

class FTRL{
    public:
        FTRL(Load_Data* load_data, Predict* predict, int total_num_proc, int my_rank) 
            : data(load_data), pred(predict), num_proc(total_num_proc), rank(my_rank){
            init();
        }
        ~FTRL(){}

        void init(){
            v_dim = data->factor * data->glo_fea_dim * data->field;
            if(data->fm == true) v_dim = data->factor * data->glo_fea_dim * 1;
            if(data->lr == true) v_dim = 0;
            loc_w = new double[data->glo_fea_dim]();
            loc_g = new double[data->glo_fea_dim]();
            glo_g = new double[data->glo_fea_dim]();
            loc_sigma = new double[data->glo_fea_dim]();
            loc_n = new double[data->glo_fea_dim]();
            loc_z = new double[data->glo_fea_dim]();

            loc_v = new double[v_dim]();
            for(int i = 0; i < v_dim; i++){
                loc_v[i] = gaussrand();
                //std::cout<<"loc_v[i]"<<loc_v[i]<<std::endl;
            }
            loc_g_v = new double[v_dim]();
            glo_g_v = new double[v_dim];
            loc_sigma_v = new double[v_dim]();
            loc_n_v = new double[v_dim]();
            loc_z_v = new double[v_dim]();

            for(int i = 0; i < data->field; i++){
                std::set<int> s;
                for(int j = 0; j < data->field; j += 1){
                    s.insert(j);
                }
                cross_field.push_back(s);
            }

             
            alpha_v = 1.0;
            beta_v = 0.01;
            lambda1_v = 0.0001;
            lambda2_v = 0.0;
        }

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

        void update_w(){// only for master node
            for(int col = 0; col < data->glo_fea_dim; col++){
                loc_sigma[col] = ( sqrt (loc_n[col] + glo_g[col] * glo_g[col]) - sqrt(loc_n[col]) ) / alpha;
                loc_n[col] += glo_g[col] * glo_g[col];
                loc_z[col] += glo_g[col] - loc_sigma[col] * loc_w[col];
                if(abs(loc_z[col]) <= lambda1){
                    loc_w[col] = 0.0;
                }
                else{
                    float tmpr= 0.0;
                    if(loc_z[col] >= 0) tmpr = loc_z[col] - lambda1;
                    else tmpr = loc_z[col] + lambda1;
                    float tmpl = -1 * ( ( beta + sqrt(loc_n[col]) ) / alpha  + lambda2);
                    loc_w[col] = tmpr / tmpl;
                }
            }//end for
        }

        void update_v_ftrl(){// only for master node
            for(int k = 0; k < data->factor; k++){
                if(data->lr == true) break;
                for(int col = 0; col < data->glo_fea_dim; col++){
                    for(int f = 0; f < data->field; f++){
                        if(data->fm == true) f = 0;
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
                        if(data->fm == true) break;
                    }
                }//end for
            }//end for
        }

        void update_v_sgd(){// only for master node
            //print2dim(glo_g_v, data->factor, data->glo_fea_dim);
            for(int k = 0; k < data->factor; k++){
                if(data->lr == true) break;
                for(int col = 0; col < data->glo_fea_dim; col++){
                    for(int f = 0; f < data->field; f++){
                        if(data->fm == true) f = 0;
                        addVal(loc_v, 1 * 0.01 *  getElem(glo_g_v, k, col, f), k, col, f);
                        if(data->fm == true) break;
                    }
                }
            }//end for
        }

        inline double getElem(double* arr, int i, int j, int k){
            return arr[i * data->glo_fea_dim*data->field + j * data->field + k];    
        }
        
        inline void putVal(double* arr, float val, int i, int j, int k){
            arr[i*data->glo_fea_dim*data->field + j * data->field + k] = val;
        }

        inline void addVal(double* arr, int val, int i, int j, int k){
            arr[i * data->glo_fea_dim*data->field + j * data->field + k] += val;
        }

        void batch_gradient_calculate(int &row){
            int group = 0, index = 0; float value = 0.0; float pctr = 0;
            for(int line = 0; line < batch_size; line++){
                float wx = bias;
                int ins_seg_num = data->fea_matrix[row].size();
                std::vector<float> vx_sum(data->factor, 0.0);
                float vxvx = 0.0, vvxx = 0.0;
                std::set<int>::iterator setIter;
                for(int col = 0; col < ins_seg_num; col++){//for one instance
                    group = data->fea_matrix[row][col].group;
                    index = data->fea_matrix[row][col].idx;
                    value = data->fea_matrix[row][col].val;
                    wx += loc_w[index] * value;
                    for(int k = 0; k < data->factor; k++){
                        if(data->lr == true) break;
                        for(int f = 0; f < data->field; f++){
                            setIter = cross_field[group].find(f);
                            if(setIter == cross_field[group].end()) continue;
                            if(data->fm == true) f = 0;
                            int loc_v_temp = getElem(loc_v, k, index, f);
                            vx_sum[k] += loc_v_temp * value;
                            vvxx += loc_v_temp * loc_v_temp * value * value;
                            if(data->fm == true) break;
                        }
                    }
                }//end for
                for(int k = 0; k < data->factor; k++){
                    if(data->lr == true) break;
                    vxvx += vx_sum[k] * vx_sum[k]; 
                }
                vxvx -= vvxx;
                wx += vxvx * 1.0 / 2.0;
                pctr = sigmoid(wx);
                float delta = pctr - data->label[row];

                for(int col = 0; col < ins_seg_num; col++){
                    group = data->fea_matrix[row][col].group;
                    index = data->fea_matrix[row][col].idx;
                    value = data->fea_matrix[row][col].val;
                    loc_g[index] += delta * value;
                    float vx = 0.0;
                    for(int k = 0; k < data->factor; k++){
                        if(data->lr == true) break;
                        for(int f = 0; f < data->field; f++){
                            setIter = cross_field[group].find(f);
                            if(setIter == cross_field[group].end()) continue;
                            if(data->fm == true) f = 0;
                            float tmpv = getElem(loc_v, k, index, f);
                            vx = tmpv * value;
                            addVal(loc_g_v, -1 * delta * (vx_sum[k] - vx) * value, k, index, f);
                            if(data->fm == true) break;
                        }
                    }
                }
                row++;
            }//end for
        }//end batch_gradient_calculate

        void print2dim(double** a, int m, int n){
            for(int i = 0; i < m; i++){
                for(int j = 0; j < n; j++){
                    if(a[i][j] != 0) std::cout<<a[i][j]<<" ";
                }
                std::cout<<std::endl;
            }
        } 
        void print1dim(double* a, int n){
            for(int i = 0; i < n; i++){
                if(a[i] != 0)std::cout<<a[i]<<" ";
            }
        }
        void save_model(int epoch){
            char buffer[1024];
            snprintf(buffer, 1024, "%d", epoch);
            std::string filename = buffer;
            std::ofstream md;
            md.open("./model/model_epoch" + filename + ".txt");
            if(!md.is_open()){
                std::cout<<"save model open file error: "<< std::endl;
            }
            float wi;
            for(int j = 0; j < data->glo_fea_dim; j++){
                wi = loc_w[j];
                md<< j << "\t" <<wi<<std::endl;
            }
            md.close();
        }

        void train(){
            int batch_num = data->fea_matrix.size() / batch_size, batch_num_min = 0;
            MPI_Allreduce(&batch_num, &batch_num_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            std::cout<<"total epochs = "<<epochs<<" batch_num_min = "<<batch_num_min<<std::endl;
            for(int epoch = 0; epoch < epochs; epoch++){
                int row = 0, batches = 0;
                std::cout<<"epoch "<<epoch<<" ";
                pred->run(loc_w, loc_v);
                if(rank == 0 && (epoch+1) % 20 == 0) save_model(epoch);
                while(row < data->fea_matrix.size()){
                    if( (batches == batch_num_min - 1) ) break;
                    batch_gradient_calculate(row);
                    if(row % 50000 == 0) std::cout<<"row = "<<row<<std::endl;
                    cblas_dscal(data->glo_fea_dim, 1.0/batch_size, loc_g, 1);
                    cblas_dscal(v_dim, 1.0/batch_size, loc_g_v, 1);

                    if(rank != 0){//slave nodes send gradient to master node;
                        MPI_Send(loc_g, data->glo_fea_dim, MPI_DOUBLE, 0, 99, MPI_COMM_WORLD);
                        MPI_Send(loc_g_v, v_dim, MPI_DOUBLE, 0, 399, MPI_COMM_WORLD);
                    }
                    else if(rank == 0){//rank 0 is master node
                        cblas_dcopy(data->glo_fea_dim, loc_g, 1, glo_g, 1);
                        cblas_dcopy(v_dim, loc_g_v, 1, glo_g_v, 1);
                        for(int r = 1; r < num_proc; r++){//receive other node`s gradient and store to glo_g;
                            MPI_Recv(loc_g, data->glo_fea_dim, MPI_DOUBLE, r, 99, MPI_COMM_WORLD, &status);
                            cblas_daxpy(data->glo_fea_dim, 1, loc_g, 1, glo_g, 1);

                            MPI_Recv(loc_g_v, v_dim, MPI_DOUBLE, r, 399, MPI_COMM_WORLD, &status);
                            cblas_daxpy(v_dim, 1, loc_g_v, 1, glo_g_v, 1);
                        }
                        cblas_dscal(data->glo_fea_dim, 1.0/num_proc, glo_g, 1);
                        cblas_dscal(v_dim, 1.0/num_proc, glo_g_v, 1);
                        update_w();
                        //print1dim(loc_w, data->glo_fea_dim);
                        //update_v_sgd();
                        update_v_ftrl();
                        //print1dim(loc_v, v_dim);
                    }
                    //sync w of all nodes in cluster
                    if(rank == 0){
                        for(int r = 1; r < num_proc; r++){
                            MPI_Send(loc_w, data->glo_fea_dim, MPI_DOUBLE, r, 999, MPI_COMM_WORLD);
                            MPI_Send(loc_v, v_dim, MPI_DOUBLE, r, 3999, MPI_COMM_WORLD);
                        }
                    }
                    else if(rank != 0){
                        MPI_Recv(loc_w, data->glo_fea_dim, MPI_DOUBLE, 0, 999, MPI_COMM_WORLD, &status);
                        MPI_Recv(loc_v, v_dim, MPI_DOUBLE, 0, 3999, MPI_COMM_WORLD, &status);
                    }
                    MPI_Barrier(MPI_COMM_WORLD);//will it make the procedure slowly? is it necessary?
                    batches++;
                }//end row while
                //print2dim(loc_g_v, data->factor, data->glo_fea_dim);
            }//end epoch for
        }//end train

    public:
        int v_dim;

        double* loc_w;
        double* loc_v;
        int epochs;
        int batch_size;

        float bias;
        float alpha;
        float alpha_v;
        float beta;
        float beta_v;
        float lambda1;
        float lambda1_v;
        float lambda2;
        float lambda2_v;
    private:
        MPI_Status status;

        std::vector<std::set<int> > cross_field;
        Load_Data* data;
        Predict* pred;
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

        int num_proc;
        int rank;
};
#endif
