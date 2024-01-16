# NOODLE: Uncertainty-Aware Hardware Trojan Detection Using Multimodal Deep Learning. 
This repository contains various components related to the research paper and source codes of the multimodal hardware Trojan detection framework.  </br>
[Rahul Vishwakarma](https://github.com/rahvis) & [Amin Rezaei](https://github.com/r3zaei) </br>
### Abstract

The risk of hardware Trojans being inserted at various stages of chip production has increased in a zero-trust fabless era. To counter this, various machine learning solutions have been developed for the detection of hardware Trojans. While most of the focus has been on either a statistical or deep learning approach, the limited number of Trojan-infected benchmarks affects the detection accuracy and restricts the possibility of detecting zero-day Trojans. To close the gap, we first employ generative adversarial networks to amplify our data in two alternative representation modalities: a graph and a tabular, which ensure a representative distribution of the dataset. Further, we propose a multimodal deep learning approach to detect hardware Trojans and evaluate the results from both early fusion and late fusion strategies. We also estimate the uncertainty quantification metrics of each prediction for risk-aware decision-making. The results not only validate the effectiveness of our suggested hardware Trojan detection technique but also pave the way for future studies utilizing multimodality and uncertainty quantification to tackle other hardware security problems.

## A. Dataset 
We have used two different dataset for the detection of hardware Trojans. 
### 1. Synthetic dataset
The dataste has binary label: _Trojan Free and Trojan Infected_. 
Graph dataset are represented as .json files and Tabular dataaset in .csv file <br>
The source code and results are in folder: ```synthetic-multimodal-graph_and_table```
### 2. Trust-Hub:  Chip Level Trojans 
The dataste has binary label: _Trojan Free and Trojan Infected_. <br>
#### 2.1 Graph dataset 
https://github.com/AICPS/hw2vec. <br>
The source code and results are in the folder: source
#### 2.2 Tabular dataset
Dataset Source: https://trust-hub.org/#/benchmarks/chip-level-trojan

## B. Proposed Solution 
![Solution](https://github.com/cars-lab-repo/NOODLE/assets/64368687/8e56628b-ec27-4e8a-a87f-dfc164e25dbd)

## C. Experiments
Use the below Python source code for reproducing the results in the paper. 

### 1. First we explore the evalaution metrics on Tabular dataset. 
This provides the ROC curve.
```
source/01_table.py
```

### 2. We also evalaute the metrics on Graph dataset. 
Provides the the ROC curve plots.
```
source/02_graph.py
```
### 3. Perform multimodal approach using early fusion.
Provides the brier score, accuracy, and confusion matrix. 
```
source/03_early_fusion.py
```
### 4. Perform multimodal approach using late fusion.
Provides the brier score, accuracy, and confusion matrix. 
```
source/04_late_fusion.py
```

## D. Results
<img width="1217" alt="results" src="https://github.com/cars-lab-repo/NOODLE/assets/64368687/afb62ccc-eec8-488f-8156-f2ccd8507f41">

## Contributions
- Proposing a multimodal learning approach using graph and euclidean data of the hardware circuits.
- Suggesting a model fusion approach using p-values with uncertainty quantifier.
- Addressing the critical issue of missing modalities and small dataset. 

## License
GNU General Public License v3.0

## Citation
```
@INPROCEEDINGS{NOODLE,
  author={Vishwakarma, Rahul and Rezaei, Amin},
  booktitle={Proceedings of 27th Design, Automation & Test in Europe Conference & Exhibition (DATE)}, 
  title={Uncertainty-Aware Hardware Trojan Detection Using Multimodal Deep Learning}, 
  year={2024},
  volume={},
  number={},
  pages={},
  doi={}
}

