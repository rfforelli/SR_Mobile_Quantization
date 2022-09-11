#
# https://github.com/GiuseppeDiGuglielmo/SR_Mobile_Quantization#prepare-div2k-data
# https://data.vision.ee.ethz.ch/cvl/DIV2K/
#

HIGH_RES="\
    DIV2K_train_HR.zip \
    DIV2K_valid_HR.zip"
LOW_RES_NTIRES2017="\
    DIV2K_train_LR_bicubic_X2.zip \
    DIV2K_train_LR_unknown_X2.zip \
    DIV2K_valid_LR_bicubic_X2.zip \
    DIV2K_valid_LR_unknown_X2.zip \
    DIV2K_train_LR_bicubic_X3.zip \
    DIV2K_train_LR_unknown_X3.zip \
    DIV2K_valid_LR_bicubic_X3.zip \
    DIV2K_valid_LR_unknown_X3.zip \
    DIV2K_train_LR_bicubic_X4.zip \
    DIV2K_train_LR_unknown_X4.zip \
    DIV2K_valid_LR_bicubic_X4.zip \
    DIV2K_valid_LR_unknown_X4.zip"
LOW_RES_NTIRES2018="\
    DIV2K_train_LR_x8.zip \
    DIV2K_train_LR_mild.zip \
    DIV2K_train_LR_difficult.zip \
    DIV2K_train_LR_wild.zip \
    DIV2K_valid_LR_x8.zip \
    DIV2K_valid_LR_mild.zip \
    DIV2K_valid_LR_difficult.zip \
    DIV2K_valid_LR_wild.zip"

mkdir -p DIV2K/bin
mkdir -p ZIP

for dset in $HIGH_RES
do
    echo "Download and unzip: $dset"
    wget -nv -P ZIP -c http://data.vision.ee.ethz.ch/cvl/DIV2K/$dset
    unzip -q -o ZIP/$dset -d DIV2K/bin
done

for dset in $LOW_RES_NTIRES2017
do
    echo "Download and unzip: $dset"
    wget -nv -P ZIP -c http://data.vision.ee.ethz.ch/cvl/DIV2K/$dset
    unzip -q -o ZIP/$dset -d DIV2K/bin
done

for dset in $LOW_RES_NTIRES2018
do
    echo "Download and unzip: $dset"
    wget -nv -P ZIP -c http://data.vision.ee.ethz.ch/cvl/DIV2K/$dset
    unzip -q -o ZIP/$dset -d DIV2K/bin
done
