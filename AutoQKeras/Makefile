help:
	@echo "INFO: make <tab> for targets"
.PHONY: help

pretrained_bayesian:
	TF_CPP_MIN_LOG_LEVEL=3 TF_XLA_FLAGS=--tf_xla_enable_xla_devices \
		python autoq.py --opt base7_autoqkeras.yaml --name base7_autoqkeras_D4C28_bs16ps64_lr1e-3 --scale 3  --bs 16 --ps 64 --lr 1e-3 --gpu_ids 0 --pretrained True --mode bayesian
.PHONY: pretrained_bayesian

pretrained_random:
	TF_CPP_MIN_LOG_LEVEL=3 TF_XLA_FLAGS=--tf_xla_enable_xla_devices \
		python autoq.py --opt base7_autoqkeras.yaml --name base7_autoqkeras_D4C28_bs16ps64_lr1e-3 --scale 3  --bs 16 --ps 64 --lr 1e-3 --gpu_ids 0 --pretrained True --mode random
.PHONY: pretrained_random

pretrained_hyperband:
	TF_CPP_MIN_LOG_LEVEL=3 TF_XLA_FLAGS=--tf_xla_enable_xla_devices \
		python autoq.py --opt base7_autoqkeras.yaml --name base7_autoqkeras_D4C28_bs16ps64_lr1e-3 --scale 3  --bs 16 --ps 64 --lr 1e-3 --gpu_ids 0 --pretrained True --mode hyperband
.PHONY: pretrained_hyperband

bayesian:
	TF_CPP_MIN_LOG_LEVEL=3 TF_XLA_FLAGS=--tf_xla_enable_xla_devices \
		python autoq.py --opt base7_autoqkeras.yaml --name base7_autoqkeras_D4C28_bs16ps64_lr1e-3 --scale 3  --bs 16 --ps 64 --lr 1e-3 --gpu_ids 0 --pretrained False --mode bayesian
.PHONY: bayesian

random:
	TF_CPP_MIN_LOG_LEVEL=3 TF_XLA_FLAGS=--tf_xla_enable_xla_devices \
		python autoq.py --opt base7_autoqkeras.yaml --name base7_autoqkeras_D4C28_bs16ps64_lr1e-3 --scale 3  --bs 16 --ps 64 --lr 1e-3 --gpu_ids 0 --pretrained False --mode random
.PHONY: random

hyperband:
	TF_CPP_MIN_LOG_LEVEL=3 TF_XLA_FLAGS=--tf_xla_enable_xla_devices \
		python autoq.py --opt base7_autoqkeras.yaml --name base7_autoqkeras_D4C28_bs16ps64_lr1e-3 --scale 3  --bs 16 --ps 64 --lr 1e-3 --gpu_ids 0 --pretrained False --mode hyperband
.PHONY: hyperband

