module load python/3.6.3/CUDA
conda create -n da python=3.8
conda activate da
conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch

# fairseq commit: 1a709b2a401ac8bd6d805c8a6a5f4d7f03b923ff
# git reset --hard 1a709b2a401ac8bd6d805c8a6a5f4d7f03b923ff

git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./

pip install tensorboard