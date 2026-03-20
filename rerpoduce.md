使用RIGA+测试单域测试效果：
```
nohup python main.py --mode train_DG --random_type TriD --Target_Dataset MESSIDOR_Base1 --Source_Dataset BinRushed > running_log.txt & #

./kill_trid_process.sh

##分析实验##
nohup python main_vali.py --mode train_DG --num_epochs 60 --Target_Dataset BinRushed --Source_Dataset BinRushed > running_log_domain1_semantic.txt & #
nohup python main_vali.py --mode train_DG --num_epochs 120 --Target_Dataset Magrabia --Source_Dataset Magrabia > running_log_domain2_semantic.txt & #
nohup python main_vali.py --mode train_DG --num_epochs 25 --Target_Dataset REFUGE --Source_Dataset REFUGE > running_log_domain3_semantic.txt & #
nohup python main_vali.py --mode train_DG --num_epochs 20 --Target_Dataset ORIGA --Source_Dataset ORIGA > running_log_domain4_semantic.txt & #
nohup python main_vali.py --mode train_DG --num_epochs 100 --Target_Dataset Drishti_GS --Source_Dataset Drishti_GS > running_log_domain5_semantic.txt & #
nohup python main_vali.py --mode train_DG --num_epochs 60 --Target_Dataset BinRushed --Source_Dataset BinRushed > running_log_domain1_randk.txt & #
nohup python main_vali.py --mode train_DG --num_epochs 120 --Target_Dataset Magrabia --Source_Dataset Magrabia > running_log_domain2_randk.txt & #
nohup python main_vali.py --mode train_DG --num_epochs 25 --Target_Dataset REFUGE --Source_Dataset REFUGE > running_log_domain3_randk.txt & #
nohup python main_vali.py --mode train_DG --num_epochs 20 --Target_Dataset ORIGA --Source_Dataset ORIGA > running_log_domain4_randk.txt & #
nohup python main_vali.py --mode train_DG --num_epochs 100 --Target_Dataset Drishti_GS --Source_Dataset Drishti_GS > running_log_domain5_randk.txt & #


nohup python main_vali.py --mode train_DG --num_epochs 60 --Target_Dataset BinRushed --Source_Dataset BinRushed > running_log_domain1_test11.txt & #
nohup python main_vali.py --mode train_DG --num_epochs 60 --Target_Dataset BinRushed --Source_Dataset BinRushed > running_log_domain1_maxk.txt & #
nohup python main_vali.py --mode train_DG --num_epochs 120 --Target_Dataset Magrabia --Source_Dataset Magrabia > running_log_domain2_maxk.txt & #
nohup python main_vali.py --mode train_DG --num_epochs 25 --Target_Dataset REFUGE --Source_Dataset REFUGE > running_log_domain3_maxk.txt & #
nohup python main_vali.py --mode train_DG --num_epochs 20 --Target_Dataset ORIGA --Source_Dataset ORIGA > running_log_domain4_maxk.txt & #
nohup python main_vali.py --mode train_DG --num_epochs 100 --Target_Dataset Drishti_GS --Source_Dataset Drishti_GS > running_log_domain5_maxk.txt & #

nohup python main_vali.py --mode train_DG --num_epochs 60 --Target_Dataset BinRushed --Source_Dataset BinRushed > running_log_domain1_style.txt & #
nohup python main_vali.py --mode train_DG --num_epochs 120 --Target_Dataset Magrabia --Source_Dataset Magrabia > running_log_domain2_style.txt & #
nohup python main_vali.py --mode train_DG --num_epochs 25 --Target_Dataset REFUGE --Source_Dataset REFUGE > running_log_domain3_style.txt & #
nohup python main_vali.py --mode train_DG --num_epochs 20 --Target_Dataset ORIGA --Source_Dataset ORIGA > running_log_domain4_style.txt & #
nohup python main_vali.py --mode train_DG --num_epochs 100 --Target_Dataset Drishti_GS --Source_Dataset Drishti_GS > running_log_domain5_loss2.txt & #

cd OPTIC
nohup python main_vali_multi.py --mode train_DG --Source_Dataset Magrabia REFUGE ORIGA Drishti_GS > running_log_multi.txt & #


nohup python main_vali_multi.py --mode train_DG --Source_Dataset BinRushed Magrabia REFUGE ORIGA > running_log_multi_domain5.txt & #

python main_vali_multi.py --mode single_test --load_time 20241223_205350_700724 --Target_Dataset Drishti_GS

python main_vali_multi.py --mode single_test --load_time 20241223_011714_189683 --Target_Dataset BinRushed

# Training
nohup python main_vali.py --mode train_DG --Target_Dataset BinRushed --Source_Dataset BinRushed > running_log_domain1_style.txt & #

nohup python main_vali.py --mode train_DG --num_epochs 120 --Target_Dataset Magrabia --Source_Dataset Magrabia > running_log_semantic_only_domain2.txt & #
nohup python main_vali.py --mode train_DG --num_epochs 25 --Target_Dataset REFUGE --Source_Dataset REFUGE > running_log_test.txt & #
nohup python main_vali.py --mode train_DG --num_epochs 20 --Target_Dataset ORIGA --Source_Dataset ORIGA > running_log_semantic_only_domain4.txt & #
nohup python main_vali.py --mode train_DG --num_epochs 100 --Target_Dataset Drishti_GS --Source_Dataset Drishti_GS > running_log_semantic_only_domain5.txt & #
# Testing
python inference.py --mode multi_test --load_time 20250224_161311_740517 --Target_Dataset Drishti_GS

nohup python main_vali.py --mode train_DG --random_type TriD --Target_Dataset REFUGE --Source_Dataset REFUGE > running_log_domain3.txt &

---------Trid----------
nohup python main.py --mode train_DG --random_type TriD --Target_Dataset BinRushed --Source_Dataset BinRushed > running_log_domain1.txt &

nohup python main.py --mode train_DG --random_type TriD --Target_Dataset Magrabia --Source_Dataset Magrabia > running_log_domain2.txt &

nohup python main.py --mode train_DG --random_type TriD --Target_Dataset REFUGE --Source_Dataset REFUGE > running_log_domain3.txt &

nohup python main.py --mode train_DG --random_type TriD --Target_Dataset ORIGA --Source_Dataset ORIGA > running_log_domain4.txt &

nohup python main.py --mode train_DG --random_type TriD --Target_Dataset Drishti_GS --Source_Dataset Drishti_GS > running_log_domain5.txt &
-----------------------

---------Mixstyle----------
nohup python main.py --mode train_DG --random_type MixStyle --Target_Dataset BinRushed --Source_Dataset BinRushed > running_log_domain1_MixStyle.txt &

nohup python main.py --mode train_DG --random_type MixStyle --Target_Dataset Magrabia --Source_Dataset Magrabia > running_log_domain2_MixStyle.txt &

nohup python main.py --mode train_DG --random_type MixStyle --Target_Dataset REFUGE --Source_Dataset REFUGE > running_log_domain3_MixStyle.txt &

nohup python main.py --mode train_DG --random_type MixStyle --Target_Dataset ORIGA --Source_Dataset ORIGA > running_log_domain4_MixStyle.txt &

nohup python main.py --mode train_DG --random_type MixStyle --Target_Dataset Drishti_GS --Source_Dataset Drishti_GS > running_log_domain5_MixStyle.txt &
-----------------------

nohup python main_vali.py --mode train_DG --Target_Dataset BinRushed --Source_Dataset BinRushed > running_log_99.txt &


# consume Training #########
nohup python main_vali.py --mode train_DG --consume True --Target_Dataset Magrabia --Source_Dataset Magrabia > running_log.txt &

# Testing
python inference.py --mode multi_test --load_time 20250224_152737_060930 --Target_Dataset Drishti_GS
20241128_003451_164488
python main.py --mode multi_test --load_time 20241128_003451_164488 --Target_Dataset ORIGA
seed=1234
w/ uncertainty
source domain1  target domain2
300epoch oc 0.635      od 0.828

w/o uncertainty
source domain1  target domain2
300epoch oc 0.633      od 0.830

# Test
CUDA_VISIBLE_DEVICES=0 python main.py --mode single_test --load_time TIME_OF_MODEL --Target_Dataset BinRushed
```
sh scripts/train.sh 3 7999
unimatch:mean_dice 88.69
unimatch w/ 特征去冗余(对feature1、feature2且loss的权重都为0.1): mean_dice 88.62
unimatch w/ 特征去冗余(对encoder所有feature且loss的权重都为0.2): mean_dice 88.71

unimatch w/ mixstyle(layer0 layer1):mean_dice 87.54
unimatch w/ mixstyle(layer1):mean_dice 87.59
unimatch w/ mixstyle(layer2):mean_dice 
unimatch w/ mixstyle(layer1+shuffle):mean_dice 86.99
---------------------------------------------------------------------------------------
seed:1234
source domain:RIGAPlus target domain MESSIDOR_Base1
w/o 不确定性 Test Metrics:  {'Disc_Dice': 0.9451881046192399, 'Disc_ASD': 4.238755727321371, 'Cup_Dice': 0.8214040824967319, 'Cup_ASD': 6.326077344653668}
w/ 语义不确定性+风格不确定性 （max：1.2 min：0.8）Test Metrics:  {'Disc_Dice': 0.9352452299124376, 'Disc_ASD': 4.7458778213774435, 'Cup_Dice': 0.8135689974953185, 'Cup_ASD': nan}
w/ 语义不确定性+风格不确定性 （max：1 min：0.6）Test Metrics:  {'Disc_Dice': 0.9457283599675539, 'Disc_ASD': 4.026797104094974, 'Cup_Dice': 0.8341047342412887, 'Cup_ASD': 5.552735680210158}


w/ 不确定性 （max：1 min：0.6）Test Metrics:  {'Disc_Dice': 0.9426930442310321, 'Disc_ASD': 4.211306445233349, 'Cup_Dice': 0.8308024919999674, 'Cup_ASD': 5.592540020207087}
w/ 语义不确定性+风格不确定性 （max：1.2 min：0.8）Test Metrics:  {'Disc_Dice': 0.9457283599675539, 'Disc_ASD': 4.026797104094974, 'Cup_Dice': 0.8341047342412887, 'Cup_ASD': 5.552735680210158}



