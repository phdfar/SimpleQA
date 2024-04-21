# SimpleQA

## Install packege

```
pip install --upgrade pip
pip install farm-haystack[colab,inference]
```

## Download sample GOT data
```
wget -O data.tar.gz https://www.dropbox.com/scl/fi/covq562eq11wg0xi7bxnj/data.tar.gz?rlkey=s1ax3o4odt3uoakawfv5z9wdx&st=f0ucdydr&dl=0
tar xfz data.tar.gz
```

## Build model
```
python QA.py -build --modelname got.obj --source data/build_your_first_question_answering_system 
```
## Test model
```
python QA.py -test  --modelname got.obj --query 'Who is the father of Arya Stark?'
```

# Another way

## Build model
```
from QA import QAService
QA  = QAService(modelname='got.obj',source='data/build_your_first_question_answering_system')
QA.build()
```

## Test model
```
QA.test('Who is the father of Arya Stark?')
```


