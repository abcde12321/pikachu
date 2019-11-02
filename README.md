# pikachu
 
## required python framework
skilearn, skimage, python3+, opencv

##to use

python train.py
'lr.pkl' will be created at 'model/lr.pkl'

##to test other images
python predict -m lr -f pikachu_dataset/extra_test/

## Steps:
1, keep a list all pokemon and not pokemon images (some images e.g. .gif are left out)
2, extract local binary pattern features 
3, train with logistic regression



## TODO:
k-fold, against adversarial attack