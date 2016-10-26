#include "ftrl_learner.h"

namespace dml{
void FtrlLearner::run(){
    int batch_num = data->fea_matrix.size() / param->batch_size, batch_num_min = 0;
    MPI_Allreduce(&batch_num, &batch_num_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    std::cout<<"total epochs = "<<param->epoch<<" batch_num_min = "<<batch_num_min<<std::endl;
    int core_num = std::thread::hardware_concurrency();
    std::thread threads[core_num];

    clock_t pstart, pend;
    clock_t start_time, finish_time;
    clock_t send_time, recv_time;
    ThreadPool pool(core_num);

    for(int epoch = 0; epoch < param->epoch; ++epoch){
        row = 0;
        int batches = 0;
        std::cout<<"epoch "<<epoch<<" ";
        if((epoch + 1) % 10 == 0){
            pstart = clock();
            pred->run(loc_w, loc_v);
            pend = clock();
            std::cout<<"predict time:"<<(pend - pstart) * 1.0 / CLOCKS_PER_SEC<<std::endl;
        }
        if(rank == 0 && (epoch+1) % 20 == 0) dump(epoch);
        if(param->issinglethread){
            while(row < data->fea_matrix.size()){
                if( (batches == batch_num_min - 1) ) break;
                batch_gradient_calculate();
                if(row % 200000 == 0) std::cout<<"row = "<<row<<std::endl;
                allreduce_gradient();
                allreduce_weight();
                batches++;
            }
        }
        else if(param->ismultithread){
            start_time = clock();
            for(int i = 0; i < batch_num_min; ++i){
                memset(loc_g, 0.0, param->fea_dim * sizeof(double));//notation:
                memset(loc_g_v, 0.0, v_dim * sizeof(double));//notation:
                std::cout<<"thread type :"<<param->issinglethread<<std::endl;
                int all_start = i * param->batch_size;
                int thread_batch = param->batch_size / core_num;
                for(int j = 0; j < core_num; ++j){
                    int start = all_start + j * thread_batch;
                    int end = all_start + (j + 1) * thread_batch;
                    pool.enqueue(std::bind(&FtrlLearner::batch_gradient_calculate_multithread, this, start, end));
                }
                mutex.lock();
                allreduce_gradient();
                allreduce_weight();
                mutex.unlock();
            }
            finish_time = clock();
            std::cout<<"Elasped time:"<<(finish_time - start_time) * 1.0 / CLOCKS_PER_SEC<<std::endl;
        }
    }
}//end run

void FtrlLearner::batch_gradient_calculate(){
    int group = 0, index = 0; float value = 0.0, pctr = 0.0;
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

void FtrlLearner::batch_gradient_calculate_multithread(int start, int end){
    int group = 0, index = 0; float value = 0.0, pctr = 0.0;
    memset(loc_g_tmp, 0.0, sizeof(double) * param->fea_dim);
    if(!param->islr) memset(loc_g_v_tmp, 0.0, sizeof(double) * v_dim);
    for(int r = start; r < end; ++r){
        float wx = bias;
        int ins_seg_num = data->fea_matrix[r].size();
        std::vector<float> vx_sum(param->factor, 0.0);
        float vxvx = 0.0, vvxx = 0.0;
        std::set<int>::iterator setIter;
        for(int col = 0; col < ins_seg_num; ++col){//for one instance
            group = data->fea_matrix[r][col].fgid;
            index = data->fea_matrix[r][col].fid;
            value = data->fea_matrix[r][col].val;
            wx += loc_w[index] * value;
            for(int k = 0; k < param->factor; ++k){
                if(param->islr) break;
                for(int f = 0; f < param->group; ++f){
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
        for(int k = 0; k < param->factor; ++k){
            if(param->islr) break;
            vxvx += vx_sum[k] * vx_sum[k];
        }
        vxvx -= vvxx;
        wx += vxvx * 1.0 / 2.0;
        pctr = sigmoid(wx);
        float delta = pctr - data->label[r];
        for(int col = 0; col < ins_seg_num; ++col){
            group = data->fea_matrix[r][col].fgid;
            index = data->fea_matrix[r][col].fid;
            value = data->fea_matrix[r][col].val;
            loc_g_tmp[index] += delta * value;
            float vx = 0.0;
            for(int k = 0; k < param->factor; ++k){
                if(param->islr) break;
                for(int f = 0; f < param->group; ++f){
                    setIter = cross_field[group].find(f);
                    if(setIter == cross_field[group].end()) continue;
                    if(param->isfm) f = 0;
                    float tmpv = getElem(loc_v, k, index, f);
                    vx = tmpv * value;
                    addVal(loc_g_v_tmp, -1 * delta * (vx_sum[k] - vx) * value, k, index, f);
                    if(param->isfm) break;
                }
            }
        }
    }//end for
    mutex.lock();
    cblas_dcopy(param->fea_dim, loc_g_tmp, 1, loc_g, 1);
    if(!param->islr)cblas_dcopy(v_dim, loc_g_v_tmp, 1, loc_g_v, 1);
    mutex.unlock();
}

void FtrlLearner::allreduce_gradient(){
    cblas_dscal(param->fea_dim, 1.0/param->batch_size, loc_g, 1);
    loc_g_nonzero = filter(loc_g, param->fea_dim);//
    std::vector<Send_datatype> loc_g_vec;
    filter_nonzero(loc_g, param->fea_dim, loc_g_vec);//

    std::vector<Send_datatype> loc_g_v_vec;
    if(!param->islr){
        cblas_dscal(v_dim, 1.0/param->batch_size, loc_g_v, 1);
        loc_g_v_nonzero = filter(loc_g_v, v_dim);//
        filter_nonzero(loc_g_v, v_dim, loc_g_v_vec);//
    }
    if(rank != 0){
        MPI_Send(&loc_g_vec[0], loc_g_nonzero, newType, 0, 99, MPI_COMM_WORLD);
        if(!param->islr) MPI_Send(&loc_g_v_vec[0], loc_g_v_nonzero, newType, 0, 399, MPI_COMM_WORLD);
    }else if(rank == 0){
        cblas_dcopy(param->fea_dim, loc_g, 1, glo_g, 1);
        for(int r = 1; r < nproc; ++r){
            std::vector<Send_datatype> recv_loc_g_vec;
            recv_loc_g_vec.resize(param->fea_dim);
            MPI_Recv(&recv_loc_g_vec[0], param->fea_dim, newType, r, 99, MPI_COMM_WORLD, &status);
            int recv_loc_g_num;
            MPI_Get_count(&status, newType, &recv_loc_g_num);
#pragma omp parallel for
            for(int i = 0; i < recv_loc_g_num; ++i){
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
#pragma omp parallel for
                for(int i = 0; i < recv_loc_g_v_num; ++i){
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
}//end allreduce_gradient

void FtrlLearner::allreduce_weight(){
    if(rank == 0){
        int loc_w_nonzero = filter(loc_w, param->fea_dim);//
        std::vector<Send_datatype> loc_w_vec;
        filter_nonzero(loc_w, param->fea_dim, loc_w_vec);//

        std::vector<Send_datatype> loc_v_vec;
        int loc_v_nonzero;
        if(!param->islr){
            loc_v_nonzero = filter(loc_v, v_dim);//
            filter_nonzero(loc_v, v_dim, loc_v_vec);//
        }
        for(int r = 1; r < nproc; ++r){
            MPI_Send(&loc_w_vec[0], loc_w_nonzero, newType, r, 999, MPI_COMM_WORLD);
            if(!param->islr) MPI_Send(&loc_v_vec[0], loc_v_nonzero, newType, r, 3999, MPI_COMM_WORLD);
        }
    }else if(rank != 0){
        std::vector<Send_datatype> recv_loc_w_vec;
        recv_loc_w_vec.resize(param->fea_dim);
        MPI_Recv(&recv_loc_w_vec[0], param->fea_dim, newType, 0, 999, MPI_COMM_WORLD, &status);
        int recv_loc_w_num;
        MPI_Get_count(&status, newType, &recv_loc_w_num);
        memset(loc_w, 0.0, param->fea_dim * sizeof(double));
#pragma omp parallel for
        for(int i = 0; i < recv_loc_w_num; ++i){
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
#pragma omp parallel for
            for(int i = 0; i < recv_loc_v_num; ++i){
                int k = recv_loc_v_vec[i].key;
                int v = recv_loc_v_vec[i].val;
                loc_v[k] = v;
            }
        }
    }
}//end allreduce_weight;

}
