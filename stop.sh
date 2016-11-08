ps -ef | grep ffm_mpi | awk '{ print $2 }' | sudo xargs kill -9
