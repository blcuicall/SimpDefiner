# Multitasking Framework for Unsupervised Simple Definition Generation

Source code for the paper **Multitasking Framework for Unsupervised Simple Definition Generation** published on **ACL 2022**.

## Requirements
### Training Environment
- Pytorch
- fairseq
- blingfire

In order to install them, you can run this command:

```
pip install -r requirements-train.txt
```

### Evaluation Environment
- Pytorch
- Sentence-Transformers
- Jieba
- NLTK
- Pandas
- scipy
- xlrd
- EASSE

In order to install them, you can run this command:

```
pip install -r requirements-eval.txt
git clone https://github.com/feralvam/easse.git
cd easse
pip install .
```

## Usage
1. All data including the Chinese and English DG dataset, and the simple text corpora mentioned in the paper have been placed in the folder "data".

2. Please download the pretrained model parameters of MASS from \[[en](https://modelrelease.blob.core.windows.net/mass/mass-base-uncased.tar.gz)|[zh](https://stublcuedu-my.sharepoint.com/:u:/g/personal/201921296062_stu_blcu_edu_cn/EZpcGUWQanxAt0XZNWb6QqsBauh4dqaR0JdF5u8ia5zJIQ?e=X2tV8r)], unzip it, and put the unzipped files into the folder "pretrained_model/MASS" and "pretrained_model/MASS-zh" respectively.

3. To preprocess the dataset, please run the following command:
```shell
bash run/data_process.sh #for English
# or
bash run/data_process_zh.sh # for Chinese
```

4. To train a SimpDefiner that can simultaneously generated complex and simple definitions, you can run the following command:
```shell
bash run/train_oxford_oald_multi_task.sh # for English
# or
bash run/train_cwn_textbook_multi_task.sh # for Chinese
```
Model checkpoints will be saved in a `checkpoint` dir.

5. If you want to evaluate the trained model and generate definitions (both complex and simple) using this model, please run the following command:

```shell
bash run/evaluate_oxford_oald.sh --model_dir [model-dir] # for English
# or
bash run/evaluate_cwn_textbook.sh --model_dir [model-dir] # for Chinese
```
The generated definitions will be saved in the same `checkpoint` dir.

6. If you want to run automatic metrics for the generated definitions, please run the following command:
```shell
bash metrics/calc_metrics.sh [model-dir] [oxford|oald|cwn|textbook] [GPU_ID]
```
The `[oxford|oald|cwn|textbook]` arguments are used to assign the specific definitions, where `[oxford|oald]` are for English, and `[cwn|textbook]` are for Chinese.

## Cite

```
@inproceedings{kong-etal-2022-simpdefiner,
    title = "Multitasking Framework for Unsupervised Simple Definition Generation",
    author = "Kong, Cunliang and
      Chen, Yun and
      Zhang, Hengyuan and
      Yang, Liner and
      Yang, Erhong",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics",
    year = "2022"
}
```

