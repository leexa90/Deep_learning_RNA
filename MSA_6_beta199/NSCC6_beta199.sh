  #!/bin/sh                                                                                                              
  #PBS -q normal                                                                                                         
  #PBS -l walltime=23:59:00                                                                                              
                                                                                                  
  
#cd /home/users/astar/bii/leexa/machine_learning
cd $PBS_O_WORKDIR
module load tensorflow/1.0

#module load python/2.7.11
python  Tensorflow_lim5000_proper_split_reweight_loss_batch.py





