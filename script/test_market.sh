export PYTHONPATH=$PYTHONPATH:./
python ./example/iids_tnorm_self_kd.py  --dataset market1501  --checkpoint /home/roya/IIDS/logs_inter/checkpoint_stage2_0_0.pth.tar  --data-dir  /home/roya/data  --evaluate 
#python ./example/iids_tnorm_self_kd.py  --dataset market1501  --checkpoint  /home/roya/IIDS/logs_intra_iids/checkpoint_stage1_0_0.pth.tar  --data-dir  /home/roya/data  --evaluate 
#python ./example/iids_tnorm_self_kd.py  --dataset market1501   --checkpoint /home/roya/IIDS/logs_intra_iids/model_best.pth.tar  --data-dir  /home/roya/data  --evaluate 
#python ./example/iids_tnorm_self_kd.py  --dataset market1501  --checkpoint  /home/roya/IIDS/logs/model_best.pth.tar  --data-dir  /home/roya/data  --evaluate 

#python ./example/iids_tnorm_self_kd.py  --dataset market1501   --checkpoint /home/roya/IIDS/logs_inter/model_best.pth.tar  --data-dir  /home/roya/data  --evaluate 

#python ./example/iids_tnorm_self_kd.py  --dataset market1501   --checkpoint /home/roya/IIDS/logs_inter/checkpoint_stage2_0_0.pth.tar  --data-dir  /home/roya/data  --evaluate 

#python ./example/iids_tnorm_self_kd.py  --dataset market1501   --checkpoint /home/roya/IIDS/logs_Intra_inter_withoutJaccard/checkpoint_stage1_0_0.pth.tar  --data-dir  /home/roya/data  --evaluate 

#python ./example/iids_tnorm_self_kd.py  --dataset market1501   --checkpoint /home/roya/IIDS/logs_Intra_inter_withoutJaccard/model_best.pth.tar  --data-dir  /home/roya/data  --evaluate 
