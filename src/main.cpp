#include <string>
#include "mpi.h"
#include "load_data.h"
#include "ftrl.h"
#include "predict.h"

int main(int argc,char* argv[]){  
    int rank, nproc;
    int namelen = 1024;
    char processor_name[namelen];
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&nproc);
    MPI_Get_processor_name(processor_name,&namelen);
    std::cout<<"my host = "<<processor_name<<" my rank = "<<rank<<std::endl;
    
    int epochnum;
    sscanf(argv[2], "epoch=%d", &epochnum);
    int batchsize;
    sscanf(argv[3], "batch_size=%d", &batchsize);
    float bias;
    sscanf(argv[4], "bias=%f", &bias);
    float alpha;
    sscanf(argv[5], "alpha=%f", &alpha);
    float beta;
    sscanf(argv[6], "beta=%f", &beta);
    float lambda1;
    sscanf(argv[7], "lambda1=%f", &lambda1);
    float lambda2;
    sscanf(argv[8], "lambda2=%f", &lambda2);
    int factor;
    sscanf(argv[9], "factor=%d", &factor);
    int group;
    sscanf(argv[10], "group=%d", &group);
    int isffm;
    sscanf(argv[11], "isffm=%d", &isffm);
    int isfm;
    sscanf(argv[12], "isfm=%d", &isfm);
    int islr;
    sscanf(argv[13], "islr=%d", &islr);

    char train_data_path[1024];
    snprintf(train_data_path, 1024, "%s-%05d", argv[14], rank);
    char test_data_path[1024];
    snprintf(test_data_path, 1024, "%s-%05d", argv[15], rank);
    if(rank == 0)std::cout<<"epochnum="<<epochnum<<"\tbatchsize="<<batchsize<<"\tbias="<<bias<<"\talpha="<<alpha<<"\tbeta="<<beta<<"\tlambda1="<<lambda1<<"\tlambda2="<<lambda2<<"\tfactor="<<factor<<"\tgroup="<<group<<"\tisffm="<<isffm<<"\tisfm="<<isfm<<"\tislr="<<islr<<std::endl;

    Load_Data test_data(test_data_path, factor, group, isffm, isfm, islr);
    test_data.load_data_batch(nproc, rank);

    Predict predict(&test_data, nproc, rank);

    Load_Data train_data(train_data_path, factor, group, isffm, isfm, islr); 
    train_data.load_data_batch(nproc, rank);

    if(strcmp(argv[1], "ftrl") == 0){
        FTRL ftrl(&train_data, &predict, nproc, rank);
        ftrl.epochs = epochnum;
        ftrl.batch_size = batchsize;
        ftrl.bias = bias;
        ftrl.alpha = alpha;
        ftrl.beta = beta;
        ftrl.lambda1 = lambda1;
        ftrl.lambda2 = lambda2;
        ftrl.train();
        std::cout<<"rank "<<rank<<" train finish!"<<processor_name<<std::endl;
        predict.run(ftrl.loc_w, ftrl.loc_v);
    }

    MPI::Finalize();
    return 0;
}
