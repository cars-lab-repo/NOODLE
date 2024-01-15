# NOODLE: Uncertainty-Aware Hardware Trojan Detection Using Multimodal Deep Learning. 

## A. Dataset 
We have used two different dataset for the detection of hardware trojans. 

### 1. Synthetic dataset
The dataste has binary label: _Trojan Free and Trojan Infected_. 
Graph dataset are represented as .json files and Tabular dataaset in .csv file <br>
The source code and results are in folder: ```synthetic-multimodal-graph_and_table```

### 2. Trust-Hub:  Chip Level Trojans 
The dataste has binary label: _Trojan Free and Trojan Infected_. <br>

#### 2.1 Graph dataset 
<br>
https://github.com/AICPS/hw2vec. <br>
The source code and results are in the folder: source

#### 2.2 Tabular dataset
<br>
Dataset Source: https://trust-hub.org/#/benchmarks/chip-level-trojan.

## B. Proposed Solution 

![BigPicture](https://github.com/rahvis/DATE2024/assets/64368687/6822ba20-e9e0-4b66-b7e0-b0fdf30fe81e)

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
<img width="1217" alt="Screenshot 2023-09-21 at 10 46 38 AM" src="https://github.com/rahvis/DATE2024/assets/64368687/9eff7212-be20-42e9-a723-c338ae07749c">

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
  booktitle={2024 Design, Automation and Test in Europe Conference |
The European Event for Electronic System Design & Test}, 
  title={NOODLE: Uncertainty-Aware Hardware Trojan Detection Using Multimodal Deep Learning}, 
  year={2024},
  volume={},
  number={},
  pages={},
  doi={}
}

