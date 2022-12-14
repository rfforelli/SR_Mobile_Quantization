help:
	@echo "INFO: make <tab> for targets"
.PHONY: help

keras_tuner:
	TF_CPP_MIN_LOG_LEVEL=3 TF_XLA_FLAGS=--tf_xla_enable_xla_devices \
		python tuner.py --opt base7_keras_tuner.yaml --name base7_keras_tuner_D4C28_bs16ps64_lr1e-3 --scale 3  --bs 16 --ps 64 --lr 1e-3 --gpu_ids 0
.PHONY: keras_tuner

