#include <string>
#include "param.h"
#include "io/load_all_data.h"
#include "learner/ftrl_learner.h"
#include "predict.h"
#include "mpi.h"

int main(int argc,char* argv[]){  
    int rank, nproc;
    int namelen = 1024;
    char processor_name[namelen];
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&nproc);
    MPI_Get_processor_name(processor_name,&namelen);
    DML::Param param(argc, argv);
    param.Init();
    
    std::cout<<"my hostname is "<<processor_name<<" my rank is "<<rank<<std::endl;

    char train_data_path[1024];
    snprintf(train_data_path, 1024, "%s-%05d", param.train_data_path.c_str(), rank);
    char test_data_path[1024];
    snprintf(test_data_path, 1024, "%s-%05d", param.test_data_path.c_str(), rank);
  
    DML::LOAD_ALL_DATA train_data(train_data_path, rank, nproc);
    train_data.load();
    DML::LOAD_ALL_DATA test_data(test_data_path, rank, nproc);
    test_data.load();

    DML::Predict predict(&test_data, &param, nproc, rank);

    if(param.isftrl == 1){
        DML::FTRL_learner ftrl(&train_data, &predict, &param, nproc, rank);
        ftrl.run();
        std::cout<<"rank "<<rank<<" train finish!"<<processor_name<<std::endl;
        predict.run(ftrl.loc_w, ftrl.loc_v);
    }

    MPI::Finalize();
    return 0;
}
