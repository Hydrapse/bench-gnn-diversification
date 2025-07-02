# bench-gnn-diversification

## Setup
1. Environment:
   ```shell
   conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 -c pytorch
   pip install torchmetrics==1.6.0
   
   conda install -c dglteam/label/th24_cu124 dgl
   pip install pandas
   pip install pydantic
   
   pip install torch_geometric
   pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
   pip install class_resolver==0.4.3
  
   pip install hydra-core==1.3.2
   pip install hydra-colorlog==1.2.0
   pip install hydra-optuna-sweeper==1.2.0
   pip install --upgrade sqlalchemy==1.4.46
   
   pip install scikit-learn==1.5.2
   pip install gdown==5.2.0
   pip install ogb==1.3.6
   
   conda install -c rapidsai -c conda-forge -c nvidia cugraph cuda-version=12.0
   pip install nx-cugraph-cu12 --extra-index-url https://pypi.nvidia.com
   
   conda install pytorch::faiss-gpu
   ```
2. Data Files:
   - The dataset will be stored under `<Project Root>/DATA`
   - The model logits will be stored under `<Project Root>/processed`

## Train GNN Experts
- Train GCN on Penn94 using config `conf/exp_sota/penn94_gcn`
    ```shell
    python train_gnn.py +exp_sota=penn94_gcn gpu=0 
    ```
- Train domain experts by partitioning datasets
    ```shell
    python train_gnn.py +exp_sota=penn94_gcn ++dataset.group=homophily ++dataset.num_experts=2 ++dataset.fraction=0.5 gpu=2
    ```
Other useful commands
- Specify directory `trial_dir=YOUR_TRIAL_NAME` when saving logits `log_logit=true`
- Print results only: `log_level=CRITICAL`
- Perform grid search (add flag `-m`):
    ```shell
    python train_gnn.py -m +exp_gnn=penn94_gcn model.conv_layers=1,2,3 model.hidden_dim=32,64 gpu=0
    ```

## Evaluation 
Evaluate Diversity, Complementarity, and Ensemble/MoE Performance for a set of specified experts
```shell
python collect_moscat.py +exp_moscat=amazon_ratings_gcn expert='[EXPERT_1_LOGIT_FILENAME,EXPERT_2_LOGITS_FILENAME]' trial_dir=YOUR_TRIAL_NAME
```