help:
	@echo "INFO: make <tab> for targets"
.PHONY: help

autoq:
	TF_CPP_MIN_LOG_LEVEL=3 TF_XLA_FLAGS=--tf_xla_enable_xla_devices \
		python autoq.py --opt base7_autoqkeras.yaml --name base7_autoqkeras_D4C28_bs16ps64_lr1e-3 --scale 3  --bs 16 --ps 64 --lr 1e-3 --gpu_ids 0
.PHONY: autoq

