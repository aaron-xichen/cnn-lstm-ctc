#!/bin/bash

raw_data_root=./dataset/english_sentence # image_root_folder
boxes_dir=$raw_data_root/boxes
images_dir=$raw_data_root/images

trans_dir=$raw_data_root/trans
split_images_dir=$raw_data_root/split_tiny_images

# convert first
if [ -d $trans_dir ];then
    echo "detect $trans_dir already exist, try to rebuild it"
    rm -rf $trans_dir
fi
mkdir $trans_dir
for file in `ls $boxes_dir`;
do
    iconv -f gb18030 -t utf8 $boxes_dir/$file > $trans_dir/$file
done

# begin to generate data
echo "generating data..."
if [ -d $split_images_dir ]; then
   echo "detect $split_images_dir already exist, try to rebuild it"
   rm -rf $split_images_dir
fi
mkdir $split_images_dir
python dataset/get_data.py $raw_data_root

# removing temporal file
echo "removing $trans_dir"
rm -rf $trans_dir
