help:
	@echo "INFO: make <tab> for targets"
.PHONY: help

test01:
	@TF_CPP_MIN_LOG_LEVEL=3 TF_XLA_FLAGS=--tf_xla_enable_xla_devices \
	python test01.py
.PHONY: test01

test02:
	@TF_CPP_MIN_LOG_LEVEL=3 TF_XLA_FLAGS=--tf_xla_enable_xla_devices \
	python test02.py
.PHONY: test02

clean-test01:
	@rm -f test01_*jpg
	@rm -f test01_*npy
.PHONY: clean-test01

clean-test02:
	@rm -f test02_*jpg
	@rm -f test02_*npy
.PHONY: clean-test02

clean: clean-test01 clean-test02
.PHONY: clean
