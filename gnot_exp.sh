
# virtualenv -p python3.11 venv
conda create --name myenv python=3.11
conda activate myenv
pip install -r requirements.txt

# remove the comment symbol to run the code in the background
conda deactivate
conda remove --n myenv --all

### an example for training Naiver-Stokes equation on irregular domains
python train.py --gpu 0 --dataset ns2d --use-normalizer unit  --normalize_x unit --component all --comment rel2  --loss-name rel2 --epochs 500 --batch-size 4 --model-name CGPT --optimizer AdamW --weight-decay 0.00005   --lr 0.001 --lr-method cycle  --grad-clip 1000.0   --n-hidden 128 --n-layers 3  --use-tb 1  # 2>&1 & sleep 20s

### an example for training heat equation on irregular domains
python train.py --gpu 0 --dataset heat2d --use-normalizer unit  --normalize_x unit --component all --comment rel2  --loss-name rel2 --epochs 500 --batch-size 4 --model-name CGPT --optimizer AdamW --weight-decay 0.00005   --lr 0.001 --lr-method cycle  --grad-clip 1000.0   --n-hidden 128 --n-layers 3  --use-tb 1

