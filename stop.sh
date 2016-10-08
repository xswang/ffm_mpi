ps -ef | grep fm_ftrl_mpi | awk '{ print $2 }' | sudo xargs kill -9
