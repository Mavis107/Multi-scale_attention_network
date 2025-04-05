# A Multi-Scale Attention Network for Steel Surface Defect Recognition
Authors: Chin Ju Chen, Ren-Shiou Liu  
Institute of Information Management, National Cheng Kung University, Tainan, 701401, Taiwan (R.O.C.)


## Description

In the steel industry, the detection and analysis of surface defects are critical for maintaining high-quality production standards. These defects, if not identified and corrected, can lead to increased wastage and operational inefficiencies. While deep learning techniques have significantly advanced the field of automated defect detection, they often require large-scale, precisely labeled datasets and substantial computational resources, which can be challenging to obtain and manage. To overcome these limitations, this study introduces a lightweight Multi-Scale Attention Network (MSA) based on the  SqueezeNet architecture. The proposed network is designed to effectively capture and emphasize defect-relevant features across different scales of the input image. The MSA network operates by first extracting feature maps at various scales using SqueezeNet. These feature maps are then processed through a carefully designed attention module that focuses on regions most likely to contain defects. By enhancing these regions, the network improves its ability to accurately detect and classify defects. One of the key innovations of the MSA network is its ability to fuse the processed feature maps, combining the strengths of each scale to produce a comprehensive analysis of the image. This fusion process not only enhances the network’s detection accuracy but also reduces the need for extensive computational resources, making it more accessible for real-world industrial applications. Experimental results on the NEU surface defect dataset demonstrates the effectiveness of the proposed MSA network. The model achieves an impressive 99.72\% classification accuracy, significantly outperforming traditional methods. Moreover, it maintains over 92\% accuracy even when trained with reduced data, showcasing its robustness and strong generalization capabilities across various defect types of different applications.

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
    source venv/bin/activate 
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


