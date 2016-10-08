rm bigdata.te-0000*
#python split_data.py test_data_old.txt 3 testdataold
python split_data.py bigdata.te.txt 3 bigdata.te
scp bigdata.te-0000* slave1:/home/worker/xiaoshu/Field-aware-Factorization-Machine-ftrl-mpi/data
scp bigdata.te-0000* slave2:/home/worker/xiaoshu/Field-aware-Factorization-Machine-ftrl-mpi/data

rm bigdata.tr-0000*
python split_data.py bigdata.tr.txt 3 bigdata.tr
#python split_data.py test_data.txt 3 test_new
scp bigdata.tr-0000* slave1:/home/worker/xiaoshu/Field-aware-Factorization-Machine-ftrl-mpi/data
scp bigdata.tr-0000* slave2:/home/worker/xiaoshu/Field-aware-Factorization-Machine-ftrl-mpi/data
