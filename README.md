# Kaggle : VSB Power Line Fault Detection

## Introduction

### On the competition 

> Medium voltage overhead power lines run for hundreds of miles to supply power to cities. 
These great distances make it expensive to manually inspect the lines for damage that doesn't immediately lead to a power outage, such as a tree branch hitting the line or a flaw in the insulator. 
These modes of damage lead to a phenomenon known as partial discharge — an electrical discharge which does not bridge the electrodes between an insulation system completely. 
Partial discharges slowly damage the power line, so left unrepaired they will eventually lead to a power outage or start a fire.
Your challenge is to detect partial discharge patterns in signals acquired from these power lines with a new meter designed at the ENET Centre at VSB. 
Effective classifiers using this data will make it possible to continuously monitor power lines for faults.

> See https://www.kaggle.com/c/vsb-power-line-fault-detection/overview

The metric was the [Matthews Correlation Coefficient](https://en.wikipedia.org/wiki/Matthews_correlation_coefficient), as the problem was an unbalanced binary classification one. 
It expects binary inputs for both predictions and truth, and is like an f1-score on steroids.

The competition took place from  November, 6 2018 to February, 14 2019.

### Results

This competiton was the first one I really invested in. I did it solo, and ended up **38th** out of 1451.
It was a really weird competition, because results were unstable.
On the public leaderboard, I was ranked about 700th, and improvements in my local cross-validation score were not correlated to improvements in the public leaderboard.

In the end, I chose to make a model that had the best CV without using any tricks that would result in my CV improving without my LB necessary improving.
Namely : checkpointing, thresholding predictions to optimize the metric. 

The model I went with achieved 0.659 CV, 0.66718 private and 0.65206 public. 

As I worked on reproducing my results a few months later, the score I reached was 0.661 CV which scored 0.68121 on private (which ranks 14th)
and 0.63618 on public. This is how unstable the results were.

## Final solution overview

### Feature Engineering

> See notebook `feature_engineering.ipynb`
- Wavelet Denoizing
- Compute features on segments of length 5000 
  - Mean, Std, Mean + Std, Mean - Std
  - Min, Max, Min - Max, Max - Mean, Min - Mean
  - Percentiles : 0.1, 1, 5, 95, 99, 99.9 and their difference with the mean
     
### Model

> See notebook `modeling.ipynb`

I first worked with LightGBM and LSTMs, but gave up on them the last week of the competition, to go with the following idea :
Each signal was made of three phases, the idea was to consider the extracted features as pixels of an image. 
My input shape was (3, nb_segments, nb_features) with nb_segments=160 and nb_features=23 in my final submission. 
I used the most common target among the three phases as the label.

The architecture I used is a very simple CNN, well regularied. 
It was trained for 100 epochs at a 0.001 learning rate and 50 epochs at a 0.0001 learning rate.

## Repository 

- `notebooks`
  - `signal_visualization.ipynb` : Some exploration
  - `feature_engineering.ipynb` : Creation of all the features that were tried
  - `modeling.ipynb` : The model as described above 
- `input`
  - `data` : data from the competition
  - `features` : features created with the feature_engineering notebook
- `output`
  - `checkpoints` : model checkpoints
  - `submissions` : submissions for the competiton

Code should be reproductible, first download the data, then run the feature extraction on both test and train data (can take a while) and the launch the modeling kernel.

## Data

Data can be downloaded on [the official Kaggle page](https://www.kaggle.com/c/vsb-power-line-fault-detection/data).
