#!/bin/bash
process_number=3
Ip=("10.101.2.89" "10.101.2.90")
for ip in ${Ip[@]}
do
    ssh worker@$ip rm /home/worker/xiaoshu/Field-aware-Factorization-Machine-ftrl-mpi/ffm_ftrl_mpi
done
scp ffm_ftrl_mpi worker@10.101.2.89:/home/worker/xiaoshu/Field-aware-Factorization-Machine-ftrl-mpi/.
scp ffm_ftrl_mpi worker@10.101.2.90:/home/worker/xiaoshu/Field-aware-Factorization-Machine-ftrl-mpi/.
#mpirun -f ./hosts -np $process_number ././ffm_ftrl_mpi ftrl 200 500 0.0 0.1 1.0 0.001 0.0 ./data/v2v_train ./data/v2v_test
mpirun -f ./hosts -np $process_number ./ffm_ftrl_mpi ftrl epoch=10 batch_size=50 bias=0.0 alpha=0.1 beta=1.0 lambda1=0.001 lambda2=0.0 factor=2 group=68 isffm=0 isfm=0 islr=1 ./data/ffm_train ./data/ffm_test
