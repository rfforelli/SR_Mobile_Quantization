help:
	@echo "INFO: make <tab> for targets"
.PHONY: help

download:
	@./download_div2k_data.sh
.PHONY: download

rotate-crop-hr:
	@./rotate_and_crop.sh \
		DIV2K/bin/DIV2K_train_HR \
		DIV2K/bin/DIV2K_valid_HR
.PHONY: rotate-crop-hr

rotate-crop-lr:
	@./rotate_and_crop.sh \
		DIV2K/bin/DIV2K_train_LR_bicubic/X3 \
		DIV2K/bin/DIV2K_valid_LR_bicubic/X3
.PHONY: rotate-crop-lr

rotate-crop: rotate-crop-hr rotate-crop-lr
.PHONY: rotate-crop

clean:
	@rm -rf DIV2K
.PHONY: clean

ultraclean: clean
	@rm -rf ZIP
.PHONY: ultraclean
