#!/bin/bash

boxes_dir=./boxes
images_dir=./images
trans_dir=./trans
output_file_path=./data.pkl

# data.pkl already exit
if [ -f $output_file_path ];then
    echo "detect $output_file_path already exist, try to rebuild it"
    rm -rf $output_file_path
fi

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
