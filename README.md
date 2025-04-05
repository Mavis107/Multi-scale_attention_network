# A Multi-Scale Attention Network for Steel Surface Defect Recognition
Authors: Chin Ju Chen, Ren-Shiou Liu  
Institute of Information Management, National Cheng Kung University, Tainan, 701401, Taiwan (R.O.C.)


## Description

We propose a multi-scale attention network for steel surface defect classification. Experimental results show that the proposed method achieves a classification accuracy of 99.72% while reducing computational resource consumption during training on the NEU surface defect dataset, maintaining an accuracy of over 92% when classifying multiple types of defects with less data. Additionally, the model demonstrates good generalization ability across various application domains.

## Installation

### Prerequisites

- Python >= 3.8
- PyTorch >= 1.10
- CUDA (for GPU acceleration)
- Additional dependencies listed in requirements.txt

### Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/Mavis107/Multi-scale_attention_network.git
    cd Multi-scale_attention_network
    ```

2. Create a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Datasets

### 1. **NEU Surface Defect Dataset**

The [NEU Surface Defect Dataset](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database) consists of images of steel surfaces with various types of defects. It is used for training and evaluating defect detection models.

- **Directory Structure**:
    ```
    dataset/
        NEU/
            class_1/
                image_1.jpg
                image_2.jpg
                ...
            class_2/
                image_1.jpg
                ...
            ...
    ```

### 2. **Malaria Cell Images Dataset (Optional for Model Generalization)**

The [Malaria Cell Images Dataset](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria) is another dataset that can be used for evaluating model performance and testing generalization.

- **Directory Structure**:
    ```
    dataset/
        malaria_dataset/
            class_1/
                image_1.jpg
                image_2.jpg
                ...
            class_2/
                image_1.jpg
                ...
            ...
    ```

## Usage

### Training the Model
To train the model, use the following command:
```bash
python train.py \
  --data-dir ./dataset \
  --dataset NEU \
  --num_classes 6 \
  --num-epochs 25 \
  --batch-size 32 \
  --lr 0.001 \
  --log-file ./train_log.txt \
  --model-type MSA_Addition_Pool35 \
  --early-stop 10 \
  --checkpoint-path ./msa.pth
```



### Inference
Use the trained model to make predictions on new data by running the following command:

```bash
python inference.py \
  --model_path ./msa.pth \
  --model_type MSA_Addition_Pool35 \
  --num_classes 6 \
  --data-dir ./inference \
  --dataset NEU_test
```


