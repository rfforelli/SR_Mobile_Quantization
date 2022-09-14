#!/bin/bash

if [ $# -eq 0 ]; then
    echo "ERROR: Usage: $0 <IMG-DIR> [<IMG-DIR> ...]"
    exit 1
fi

IMG_DIRS=$@;

for d in $IMG_DIRS;
do
    if [ ! -d "$d" ]; then
        echo "ERROR: $IMG_DIR is not a valid directory"
        exit 1
    fi
done

for d in $IMG_DIRS;
do
    IMG_COUNT=`ls $d | wc -l`
    echo "INFO: There are $IMG_COUNT images in $d"
done


echo "INFO: Press any key to start, or CTRL+C to terminate"
read dummy

# Rotate all of the images in portrait format (height > width)
echo "INFO: Rotate (if necessary)"
for d in $IMG_DIRS;
do
    for f in $d/*.png
    do
        r=$(identify -format '%[fx:(h>w)]' "$f")
        if [[ r -eq 1 ]]
        then
            echo "INFO: Rotate $f"
            convert "$f" -rotate 90 "$f"
        fi
    done
done

# All of the images are in landscape mode now.
echo "INFO: Check width"
w_expected=-1
for d in $IMG_DIRS;
do
    for f in $d/*.png
    do
        w=$(identify -format "%w" "$f")
        if test $w_expected -eq -1; then
            w_expected=$w
        fi
        if test ! "$w_expected" -eq "$w" ; then
            echo "ERROR: Image $f has an unexpected width: $w"
            exit 1
        fi
    done
done

# At this point, the images have the same width, but they may have different
# heights.

# Find the minimum height in the dataset
echo "INFO: Check height"
h_min=16384
for d in $IMG_DIRS;
do
    for f in $d/*.png
    do
        h=$(identify -format "%h" "$f")
        if test "$h" -le "$h_min" ; then
            h_min=$h
        fi
    done
done

# Crop the images
echo "INFO: Crop all of the images to: $w_expected x $h_min"
for d in $IMG_DIRS;
do
    for f in $d/*.png
    do
        mogrify -crop $w_expected\x$h_min+0+0 $f
    done
done

# All of the images are in landscape mode and have the same dimensions.

echo "INFO: Done"
