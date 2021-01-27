prefix=bert

for dir in $prefix-clusters-sent-data $prefix-clusters-doc-data
do
    cd $dir
    
    pwd

    for subdir in '0' '1' '2' '3' 
    do

    cd $subdir
    pwd

    sbatch /home/maksym/research/da/scripts-train-nmt/ft-command-test.sh

    cd ..
    done
    cd ..

done