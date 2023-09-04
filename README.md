## LSMGA-MKGC
The repository for ACL 2023 paper: Multilingual Knowledge Graph Completion with Language-Sensitive Multi-Graph Attention

## Requirements
* python==3.6.10
* pytorch==1.10.0
* torch_gemetric==2.0.3
* torch-cluster==1.5.9
* torch-scatter==2.0.9
* torch-sparse==0.6.12

## How to run
For DBP5L dataset
```
python --run_model.py --dataset dbp5l --round 80
```
For EPKG dataset
```
python --run_model.py --dataset depkg --round 50
```
