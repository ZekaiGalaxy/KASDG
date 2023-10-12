# KASDG
Official Implementation for [Stylized Dialogue Generation with Feature-Guided Knowledge Augmentation]() (EMNLP2023 Findings).

**Introducing KASDG**: A Stylized Dialogue Response Generator that leverages style corpus from Knowledge Base perspective. **KASDG** extracts style knowledge from a style corpus and employs a uniquely designed **Feature-Guided Selection Module**, integrating **Response-Related Contrastive Learning** and **Style Responsiveness Kullback-Leibler** loss.

## Dataset

* **[Reddit](https://github.com/silverriver/Stylized_Dialog/tree/main/TCFC)** : Dialogue Corpus.

* **[TCFC](https://github.com/silverriver/Stylized_Dialog/tree/main/TCFC)** : Style corpus that contains the Formal and Informal styles.

* **[ArXiv & Holmes](https://github.com/golsun/StyleFusion)** : Style corpus that contains Holmes and ArXiv styles.

## Environment
We tested our code on CUDA 11.4.
```
cd KASDG
conda create -n kasdg python=3.10
pip install -r requirements.txt
conda activate kasdg
```

## Preparation
* **Model**: Download the pretrained weights of [bart](https://huggingface.co/facebook/bart-base) and [bart_dialogue](https://huggingface.co/tareknaous/bart-daily-dialog). Move them to `model` and name as `bart`,`bart_dialogue`, respectively.
* **Dataset** : We provide our used dataset in `data`. Same data can be downloaded from links mentioned in `Dataset` section.
* **Preprocess** : To preprocess the data and complete the retrieval process overhead `python src/prepare_data.py --dataset='your_dataset' --preprocess=1`

## Usage
* To train a model `python train.py --dataset='your_dataset'`
* To predict or test with a trained model `python eval.py --load_path='your_model_path' --load_step==your_step --dataset='your_dataset'`

## Evaluation
We provide our pretrained checkpoints through [Baidu NetDisk](https://pan.baidu.com/s/1_Bd-xpYm8txUQ9CoGyCqiQ?pwd=cwaz) or [Google Drive](https://drive.google.com/drive/folders/1_AGWVcz25nQt6KnfmffFsW9LUf8aE7oI?usp=sharing), please download them and move to `checkpoint/our_ckpt`. We use the same evaluation metrics as the original paper. For `TCFC` evaluations, please refer to [this paper](https://github.com/silverriver/Stylized_Dialog/tree/main/TCFC). For `ArXiv&Holmes` evaluations, please refer to [this paper](https://github.com/golsun/StyleFusion). You can also use your own metrics to evaluate the performance.

## Citation
If you find our work useful in your research, please consider citing:

```
```

## Contact
If you have any questions, please open a github issue or contact us:

Zekai Zhang: justinzzk@stu.pku.edu.cn



