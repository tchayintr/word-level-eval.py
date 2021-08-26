# word-level-eval.py

Python script of word-level evaluation for word segmentation task

### Usage
```
usage: word-level-eval.py [-h] [--test] [--ref_data REF_DATA] [--hyp_data HYP_DATA] [--delimiter DELIMITER] [--minimal]

evaluating tagging results using word-level criteria

optional arguments:
  -h, --help            show this help message and exit
  --test
  --ref_data REF_DATA
  --hyp_data HYP_DATA
  --delimiter DELIMITER
  --minimal             report only micro-f1 score (default: False)
```

### Output example
```
processed 21 tokens with 10 reference tokens; found: 11 hypothesis tokens; correct: 7.
accuracy:  66.67%
micro: precision:  63.64%; recall:  70.00%; FB1:  66.67
macro: precision:  66.67%; recall:  72.22%; FB1:  69.05
```
