#eval for SGPD
CUDA_VISIBLE_DEVICES='0' python imagenet_eval.py --uaps_save ./uaps_save/spgd/spgd_10000_20epoch_250batch.pth --batch_size 250 --model_name alexnet 2>&1|tee ./uaps_save/spgd/spgd_250batch_alexnet.log

