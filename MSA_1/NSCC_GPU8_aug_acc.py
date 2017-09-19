  #!/bin/sh                                                                                                              
  #PBS -q normal                                                                                                         
  #PBS -l walltime=23:59:00                                                                                              
                                                                                                  
  
#cd /home/users/astar/bii/leexa/machine_learning
cd $PBS_O_WORKDIR
module load tensorflow/1.0

#module load python/2.7.11
python data_preTF8_7_layerInception_more_param3_avg_15window_load_aug_acc.py 
#p#ython *8.py 
## CV 0.456 for test9
#python RF_v3test.py second7_train34568.csv test3456789.csv  124125
#python RF_v3test.py second8_train34569.csv test3456789.csv  3463467

#python RF_v3test.py second8_train34567.csv test3456789.csv  1254422 
#python RF_v3test.py second8_train34569.csv test3456789.csv  215252 

#python RF_v3test.py second9_train34567.csv test3456789.csv  21252 
#python RF_v3test.py second9_train34568.csv test3456789.csv  2646342 

#python RF_v3test.py second7_train34568.csv test3456789.csv  256852 
#python RF_v3test.py second8_train34569.csv test3456789.csv  225856 

#python RF_v3test.py second8_train34567.csv test3456789.csv  22543 
#python RF_v3test.py second8_train34569.csv test3456789.csv  2226 

#python RF_v3test.py second9_train34567.csv test3456789.csv  2226253 
#python RF_v3test.py second9_train34568.csv test3456789.csv  2354222




