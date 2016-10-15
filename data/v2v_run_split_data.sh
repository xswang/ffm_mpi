rm v2v_test-0000*
python split_data.py v2v_test.txt 3 v2v_test
scp v2v_test-0000* slave1:/home/worker/xiaoshu/Field-aware-Factorization-Machine-ftrl-mpi/data
scp v2v_test-0000* slave2:/home/worker/xiaoshu/Field-aware-Factorization-Machine-ftrl-mpi/data

rm v2v_train-0000*
python split_data.py v2v_train.txt 3 v2v_train
scp v2v_train-0000* slave1:/home/worker/xiaoshu/Field-aware-Factorization-Machine-ftrl-mpi/data
scp v2v_train-0000* slave2:/home/worker/xiaoshu/Field-aware-Factorization-Machine-ftrl-mpi/data
