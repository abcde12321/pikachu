# binary classification of pikachu image
 
## required python framework
skilearn, skimage, python3+, opencv

## to use

`python train.py`
'lr.pkl' will be created at 'model/lr.pkl'

#to test against other images

`python predict -m lr -f pikachu_dataset/extra_test/`

## Steps:
* keep a list all pokemon and not pokemon images (some images e.g. .gif are left out)
* extract local binary pattern features 
* train with logistic regression

## known issues
* gif image are left out
* some images are broken (libpng warning: iCCP: known incorrect sRGB profile)
* accuracy are very low (probably should use features better than local binary pattern and neural net instead of logistic regression)

## TODO:
k-fold, against adversarial attack