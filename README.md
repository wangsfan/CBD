# CBD
## Introduction
This is our implementation of our paper *Coalition Banzhaf Distillation for Continual Image-Text Retrieval*.

**TL;DR**: Coalition Banzhaf Distillation for Continual Image-Text Retrieval

**Abstract**:
In the era of big data, cross-modal retrieval is required to conduct over sequential multimedia data instead of single dataset. This leads to a new learning paradigm: continual image-text retrieval (CITR), which suffers from two challenges: cross-modal semantic discrepancy and cross-task catastrophic forgetting of knowledge. Traditional methods handle two challenges with knowledge distillation or meta-learning based on LSTM. However, the temporal memory mechanism of LSTM can jeopardize the balance of coarse-grained and fine-grained semantics, and produce severe inter-task knowledge bias. Inspired from the data valuation theory and tokenized mechanism of transformers, we regard each image token and each text token as coalition workers, and tailor for a Coalition Banzhaf valuation to characterize multi-grained semantics of image-text pairs. Based on this idea, we propose the Coalition Banzhaf Distillation Network (CBDNet) for CITR. The core component of CBDNet is the Coalition Banzhaf Distillation module, which preserves the structural cross-modal knowledge with the Coalition Banzhaf value from the cooperative game perspective, and retains the cross-task structural knowledge with a token-level similarity closeness constraint. Besides, CBDNet utilizes the dynamic semantic prompt learning and the sample-level contrastive learning and distillation, so that both cross-modal semantic and coarse-grained  knowledge are faithfully compensated. Extensive experiments on the Pascal Sentence and Wikipedia datasets demonstrate the effectiveness of CBDNet under single-set and multi-set scenarios.


## Dependencies
- decord
- pandas
- ftfy
- regex
- tqdm
- opencv-python
- functional
- timm
- boto3



## Usage

##### 1. Install dependencies
First we recommend to create a conda environment by using the following command.
```
conda create -n CBD python=3.9
```
This command creates a conda environment named `CBD`. You can activate the conda environment with the following command:
```
conda activate CBD
```
The code is written for python `3.9`, but should work for other version with some modifications.
```
pip install -r requirements.txt
pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html
```



##### 2. Run code
    ```
    python -m train_prompt <Scenario> \
        --data-path ./data/ \
        --output_dir ./output 
    ```

## Parameters

| Parameter         |           Description                       | 
|-------------------|---------------------------------------------|
| Scenario             |   Continuous learning Scenario to use        |
| incremental_steps     |   Number of tasks                                      |
| lr             |   Learning rate               |
| batch_size        |   batch size for training    |
| epochs            |   epochs                    |

## 

