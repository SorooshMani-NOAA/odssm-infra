L_DEF_DIR=~/sandbox/ondemand-storm-workflow/singularity/
L_IMG_DIR=/lustre/imgs

mkdir -p $L_IMG_DIR
for i in prep; do
    pushd $L_DEF_DIR/$i/
    sudo singularity build $L_IMG_DIR/$i.sif $i.def
    popd
done
