#!/bin/bash
conda activate digimon

python main_yago.py -opt Option/Method/Dalk.yaml -dataset_name yago

python main_yago.py -opt Option/Method/MedG.yaml -dataset_name yago # 跑不起来

python main_yago.py -opt Option/Method/ToG.yaml -dataset_name yago # 跑不起来

python main_yago.py -opt Option/Method/GR.yaml -dataset_name yago