#eval for SGA
CUDA_VISIBLE_DEVICES='0' python imagenet_eval.py --uaps_save ./uaps_save/sga/sga_10000_20epoch_250batch.pth --batch_size 100 --model_name alexnet 2>&1|tee ./uaps_save/sga/sga_250batch_alexnet.log

CUDA_VISIBLE_DEVICES='0' python imagenet_eval.py --uaps_save ./uaps_save/sga/sga_10000_20epoch_250batch.pth --batch_size 100 --model_name googlenet 2>&1|tee ./uaps_save/sga/sga_250batch_googlenet.log

CUDA_VISIBLE_DEVICES='0' python imagenet_eval.py --uaps_save ./uaps_save/sga/sga_10000_20epoch_250batch.pth --batch_size 100 --model_name vgg16 2>&1|tee ./uaps_save/sga/sga_250batch_vgg16.log

CUDA_VISIBLE_DEVICES='0' python imagenet_eval.py --uaps_save ./uaps_save/sga/sga_10000_20epoch_250batch.pth --batch_size 100 --model_name vgg19 2>&1|tee ./uaps_save/sga/sga_250batch_vgg19.log

CUDA_VISIBLE_DEVICES='0' python imagenet_eval.py --uaps_save ./uaps_save/sga/sga_10000_20epoch_250batch.pth --batch_size 100 --model_name resnet152 2>&1|tee ./uaps_save/sga/sga_250batch_resnet152.log