#!/bin/bash
if [ $# != 1 ]; then
    echo "USAGE: $0 root_dir"
    echo "e.g.: $0 /home/share/data_for_BLSTM_CTC/Samples_for_English/20151113"
    exit 1;
fi

# in_dir not exist
if [ ! -d $1 ]; then
    echo "$1 not exist, do nothing!"
    exit 1;
fi

boxes_dir=$1/boxes
images_dir=$1/images
trans_dir=$1/trans
output_file_path=$1/data.pkl

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
    iconv -f gb2312 -t utf8 $boxes_dir/$file > $trans_dir/$file
done

# invoke python script
python get_data.py $images_dir $trans_dir $output_file_path
