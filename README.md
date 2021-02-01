## requierments:

- torch
- torchvision

## How to run:

```bash
python main.py -m models/model_best.pt -p ROOT_PATH
```

This would run the best trained model on the same validation set I have used.
The ROOT_PATH should be configured to the location of the directory containing the CIFAR10 cifar-10-batches-py directory with the data_batch_x files.
If you do not have this data or it is partial, please do not pass the -p argument, and the dataset would be downloaded for you.

## Retrain

For documentation of options designed to retrain simply run:

```bash
python main.py -m models/model_best.pt
```

## Eval:

Evaluation is calculated with two metrics:
- Exact accuracy: a correct output is an output that matches correctly the two labels. In this metric a random tagger would acheive 2.22% precision.
- Partial accuracy (or multi label precision): an output that gets one label correct gets one point, and two labels correct gets two points, so the maximal points are 10000. In this metric a random tagger would acheive 20% precision.

## Score:

Exact accuracy: 19.88%
Partial accuracy: 49.7%
