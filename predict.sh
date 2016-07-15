model_path=snapshot/99.pkl
test_img_list_path=./dataset/english_sentence/test_img_list.txt
test_output_dir=./dataset/english_sentence/test_output

# check for test_output folder
if ! [ -d $test_output_dir ]; then
   echo "creating $test_output_dir"
   mkdir $test_output_dir
else
   echo "cleaning $test_output_dir"
   rm -f $test_output_dir/*
fi

# predict
python predict/predict.py $model_path $test_img_list_path $test_output_dir
