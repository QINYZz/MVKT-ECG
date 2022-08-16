set -e
:<<!
*****************Instruction*****************
Here is Pytorch training on CPSC dataset
Modify the following settings as you wish !
*********************************************
!
# short description of method
# description=baseline
description=unified_finetune_student_init_random_t_0.07
#*********************************************
alpha=1
beta=0.8
teacher_backbone=xresnet101
student_backbone=xresnet101
#*******************Dataset*******************
#Choose the dataset folder
#benchmark=PTBXL
benchmark=CPSC 
task="nothing"
#task="superdiagnostic"  
#dataset=/home/huichen/data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/
dataset="/home/huichen/data/challenge/TRAIN"
pretrained="/home/yuzhenqin/MVKT-ECG-QIN/Results_teacher/FromScratch-CPSC-nothing-resnet34-teacher-lr1e-3-bs32/model_best.pth.tar"
student_pretrain=None
#*********************************************
#****************Hyper-parameters*************
# Freeze the layers before a certain layer.
freeze_layer=0
# Batch size
batchsize=32
# The number of total epochs for training
epoch=100
# The inital learning rate
lr=1e-3
gpu_id=3,4
weight_decay=1e-4
classifier_factor=1
#*********************************************
echo "Start training!"
modeldir=Results_multi_view/FromScratch-$benchmark-$task-T-$teacher_backbone-S-$student_backbone-$description-lr$lr-bs$batchsize-alpha$alpha-beta$beta
if [ ! -d  "Results_multi_view" ]; then
mkdir Results_multi_view

fi

if [ ! -e $modeldir/*.pth.tar ]; then

if [ ! -d  "$modeldir" ]; then

mkdir $modeldir

fi
cp "transferring.sh" $modeldir
cp "transferring.py" $modeldir


python transferring.py $dataset\
               --benchmark $benchmark\
               --task $task\
               --alpha $alpha\
               --beta $beta\
               -p 50\
               --epochs $epoch\
               --lr $lr\
               --gpu $gpu_id\
               -j 4\
               -bs $batchsize\
               --freezed-layer $freeze_layer\
               --classifier-factor $classifier_factor\
               --teacher_backbone $teacher_backbone\
               --student_backbone $student_backbone\
               --benchmark $benchmark\
               --modeldir $modeldir\
               --test_per_epoch\
               --pretrained $pretrained\
               --student_pretrain $student_pretrain

else
checkpointfile=$(ls -rt $modeldir/*.pth.tar | tail -1)

python single_finetune.py $dataset\
               --benchmark $benchmark\
               --task $task\
               --_lambda $_lambda\
               --temp $temp\
               -p 10\
               --epochs $epoch\
               --lr $lr\
               --gpu $gpu_id\
               -j 4\
               -b $batchsize\
               --freezed-layer $freeze_layer\
               --modeldir $modeldir\
               --classifier-factor $classifier_factor\
               --teacher_backbone $teacher_backbone\
               --student_backbone $student_backbone\
               --benchmark $benchmark\
               --resume $checkpointfile\
               --pretrained $pretrained\
               --student_pretrain $student_pretrain


fi
echo "Done!"
