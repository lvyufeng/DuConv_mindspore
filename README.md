## 训练过程

### 用法

#### 安装GPU版本MindSpore

```bash
pip install mindspore-gpu==1.5.0
```

#### 下载和预处理数据

```bash
# download dataset
bash scripts/download_dataset.sh
# build dataset
bash scripts/build_dataset.sh [task_type]
# convert to mindrecord
bash scripts/convert_dataset.sh [task_type]
# task type: match, match_kn, match_kn_gene
# default: match_kn_gene
```

#### GPU处理器上运行

```bash
bash script/run_train.sh [task_type]
# task type: match, match_kn, match_kn_gene
# default: match_kn_gene
```
#### GPU处理器上预测
```bash
bash script/run_predict.sh
# task type: match, match_kn, match_kn_gene
# default: match_kn_gene
```
