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
    
    int epochnum = atoi(argv[2]);
    int batchsize = atoi(argv[3]);
    float bias = atof(argv[4]);
    float alpha = atof(argv[5]);
    float beta = atof(argv[6]);
    float lambda1 = atof(argv[7]);
    float lambda2 = atof(argv[8]);

    char train_data_path[1024];
    snprintf(train_data_path, 1024, "%s-%05d", argv[9], rank);
    char test_data_path[1024];
    snprintf(test_data_path, 1024, "%s-%05d", argv[10], rank);

    Load_Data test_data(test_data_path);
    test_data.load_data_batch(nproc, rank);
    Predict predict(&test_data, nproc, rank);

    Load_Data train_data(train_data_path); 
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
        ftrl.ftrl();
        std::cout<<"rank "<<rank<<" train finish!"<<processor_name<<std::endl;
        predict.run(ftrl.loc_w, ftrl.loc_v);
    }

    MPI::Finalize();
    return 0;
}
