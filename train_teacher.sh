set -e
:<<!
*****************Instruction*****************
Here is Pytorch training on PTBXL dataset
Modify the following settings as you wish !
*********************************************
!
# short description of method
description=teacher
#*********************************************

#*******************Dataset*******************
#Choose the dataset folder
benchmark=PTBXL
#benchmark=CPSC
task="superdiagnostic"
backbone=xresnet101
#dataset="/home/huichen/data/challenge/TRAIN/"
dataset="/home/huichen/data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/"
num_leads=12
#*********************************************
pretrained_model=None
# pretrained_model="/home/huichen/codes/moco_new/Results_pretrain/contrstive-learning_t_0.1_ptb_100HZ_res1d_34/model_best.pth.tar"
#****************Hyper-parameters*************
# Freeze the layers before a certain layer.
freeze_layer=0
# Batch size
batchsize=32
# The number of total epochs for training
epoch=100
# The inital learning rate
lr=1e-3
gpu_id="1,2"
weight_decay=1e-4
classifier_factor=1
#*********************************************

echo "Start training!"
modeldir=Results_teacher/FromScratch-$benchmark-$task-$backbone-$description-lr$lr-bs$batchsize
if [ ! -d  "Results_teacher" ]; then
mkdir Results_teacher

fi

if [ ! -e $modeldir/*.pth.tar ]; then

if [ ! -d  "$modeldir" ]; then

mkdir $modeldir

fi
cp "train_teacher.sh" $modeldir
cp "train_teacher.py" $modeldir


python train_teacher.py $dataset\
           --benchmark $benchmark\
           --task $task\
           -p 50\
           --epochs $epoch\
           --lr $lr\
           --gpu $gpu_id\
           -j 4\
           -b $batchsize\
           --backbone $backbone\
           --num-leads $num_leads\
           --pretrained $pretrained_model\
           --freezed-layer $freeze_layer\
           --classifier-factor $classifier_factor\
           --benchmark $benchmark\
           --test_per_epoch\
           --modeldir $modeldir

else
checkpointfile=$(ls -rt $modeldir/*.pth.tar | tail -1)

python unified_pretrain.py $dataset\  
           --benchmark $benchmark\
           --train-target $train_target\
           -p 10\
           --epochs $epoch\
           --lr $lr\
           --gpu $gpu_id\
           -j 4\
           -b $batchsize\
           --num-classes $num_classes\
           --num-leads $num_leads\
           --freezed-layer $freeze_layer\
           --modeldir $modeldir\
           --classifier-factor $classifier_factor\
           --benchmark $benchmark\
           --resume $checkpointfile
           
fi
echo "Done!"
