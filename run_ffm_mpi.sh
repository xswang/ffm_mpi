#!/bin/bash
process_number=3
Ip=("10.101.2.89" "10.101.2.90")
for ip in ${Ip[@]}
do
    ssh worker@$ip rm /home/worker/xiaoshu/Field-aware-Factorization-Machine-ftrl-mpi/ffm_ftrl_mpi
done
scp ffm_ftrl_mpi worker@10.101.2.89:/home/worker/xiaoshu/Field-aware-Factorization-Machine-ftrl-mpi/.
scp ffm_ftrl_mpi worker@10.101.2.90:/home/worker/xiaoshu/Field-aware-Factorization-Machine-ftrl-mpi/.
#mpirun -f ./hosts -np $process_number ././ffm_ftrl_mpi ftrl 200 1000 0.0 0.1 1.0 0.001 0.0 ./data/v2v_train ./data/v2v_test
mpirun -f ./hosts -np $process_number ./ffm_ftrl_mpi ftrl 10 5 0.0 0.1 1.0 0.001 0.0 ./data/ffmdata_train ./data/ffmdata_test
