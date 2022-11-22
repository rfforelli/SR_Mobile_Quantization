help:
	@echo "INFO: make <tab> for targets"
.PHONY: help

train-base7_D4C28_bs16ps64_lr1e-3:
	TF_CPP_MIN_LOG_LEVEL=3 TF_XLA_FLAGS=--tf_xla_enable_xla_devices \
		python train.py --opt options/train/base7.yaml --name base7_D4C28_bs16ps64_lr1e-3 --scale 3  --bs 16 --ps 64 --lr 1e-3 --gpu_ids 0
.PHONY: train-base7_D4C28_bs16ps64_lr1e-3

train-base7-qkeras:
	TF_CPP_MIN_LOG_LEVEL=3 TF_XLA_FLAGS=--tf_xla_enable_xla_devices \
		python train.py --opt options/train/base7_qkeras.yaml --name base7_qkeras_D4C28_bs16ps64_lr1e-3 --scale 3  --bs 16 --ps 64 --lr 1e-3 --gpu_ids 0
.PHONY: train-base7-qkeras

train-nogpu-base7_D4C28_bs16ps64_lr1e-3:
	TF_CPP_MIN_LOG_LEVEL=3 TF_XLA_FLAGS=--tf_xla_enable_xla_devices \
		python train.py --opt options/train/base7.yaml --name base7_D4C28_bs16ps64_lr1e-3 --scale 3  --bs 16 --ps 64 --lr 1e-3 --gpu_ids -1
.PHONY: train-nogpu-base7_D4C28_bs16ps64_lr1e-3

