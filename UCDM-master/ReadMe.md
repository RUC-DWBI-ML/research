# UCDM

UCDM (Unsupervised Learning for Class Distribution Mismatch) is a project that involves two main components:
1. **synthetic_data_pipeline**: Generates positive and negative data.
2. **classifier_training**: Trains a classifier using the generated data.

## Directory Structure

```
UCDM/
│
├── synthetic_data_pipeline/
│   ├── generate_positive_negative.sh  # Script to generate positive and negative data
│   ├── Dmain.py                      # Python script for generating synthetic data
│   └── ...                            # Other files related to synthetic data generation
│
├── classifier_training/
│   ├── train_classifier.sh            # Shell script to train the classifier
│   ├── Cmain.py                      # Python script to train the classifier
│   └── ...                            # Other files related to classifier training
│
└── README.md                          # Project README
```

## Setup

### 1. Install Dependencies

To run the code in this project, you need to install the following dependencies. We recommend using **Conda** to manage the environment:

```bash
conda install --yes --file requirements.txt
```
In addition, to use the **Stable Diffusion model**, you need to download the pre-trained weights from [Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-2-base):

- Model Information: [stable-diffusion-2-base](https://huggingface.co/stabilityai/stable-diffusion-2-base)
- Download the model and specify the path to `--model_path` in the `synthetic_data_pipeline` configuration.

Make sure to set up any environment variables or configurations needed by your project, such as paths to datasets and pre-trained model configurations.

### 2. Project Structure Overview

- **synthetic_data_pipeline**:
    - This folder is responsible for generating the positive and negative data. It includes scripts for creating the synthetic dataset.
    - You can customize the generation process by modifying `generate_positive_negative.sh`.
    - `Dmain.py` is the Python script that processes data generation, leveraging pre-trained models like Stable Diffusion.

- **classifier_training**:
    - This folder handles the training of the classifier using the generated synthetic data.
    - The training process can be managed by modifying the `train_classifier.sh` shell script.
    - `Cmain.py` is the main Python script for model training. It takes generated data to train a classifier.

### 3. Running the Scripts

#### Step 1: Generate Positive and Negative Data

To generate the positive and negative data, run the following shell script in the `synthetic_data_pipeline` folder:

```bash
cd synthetic_data_pipeline
sh generate_positive_negative.sh
```

This script generates the synthetic dataset, using parameters defined inside the script or specified through command-line arguments, and saves it in the designated directories. You can modify the paths and parameters inside `generate_positive_negative.sh` as needed.

#### Step 2: Train the Classifier

After generating the data, you can use it to train the classifier by running the following shell script in the `classifier_training` folder:

```bash
cd classifier_training
sh train_classifier.sh
```

This script starts the training process, using the synthetic data generated from `synthetic_data_pipeline`. The classifier will be trained with the parameters and configurations defined in the shell script.

### 4. Configuration Parameters

Both the data generation and classifier training processes are configurable via command-line arguments and configuration files.

#### `synthetic_data_pipeline` (Parameters for `generate_positive_negative.sh`):

- **Data Generation Settings**:
    - `--gpu`: Specifies the GPUs to use for distributed training.
    - `--num_gpus`: The number of GPUs to use for data generation.
    - `--save_path`: Path where the generated synthetic data will be saved.
    - `--model_path`: Path to the pre-trained model (e.g., Stable Diffusion model).
    - `--data_dir`: Directory where the raw data is stored.
    - `--data_name`: The name of the dataset (e.g., cifar10).
    - `--known_class`: List of known classes for the dataset.
    - `--unknown_class`: List of unknown categories in the unlabeled data.
    - `--new_class`: List of new categories for the test data.
    - `--test_size`: Size of the test datasets for different categories.
    - `--class_prompt`: Class descriptions used for data generation (e.g., "airplane, automobile").
    - `--batch_size`:The batch size used during training. 
    - `--batch_num`: The number of times to process a batch, typically limited by GPU capacity. 
    

#### `classifier_training` (Parameters for `train_classifier.sh`):

- **Training Settings**:
    - `--epochs`: The number of epochs for training.
    - `--batch_size`: The batch size used during training.
    - `--lr`: The learning rate for training.
    - `--seed`: Random seed for reproducibility.
    - `--select_epoch`:Specifies how many epochs to run for confidence-based labeling (default: 40).
    - `--threshold`: The confidence threshold used for labeling (default: 0.98).
    - `--mismatch`: The proportion of mismatch in the unlabeled data.
    - `--load_path`: Path to the generated dataset.
    - `--lambda1`: Strength of the class-specific function.
    - `--lambda2`: Strength of the class-specific function.
    - `--threshold`: Confidence threshold for pseudo-labeling.
    - `--config`: Path to the configuration file (e.g., `cifar10_config.json`).
    - `--data_dir`: Directory where the raw data is stored.
    - `--data_name`: The name of the dataset (e.g., cifar10).




### 5. Example Usage

Here’s an example of running the pipeline:

1. **Generate the synthetic data**:

```bash
cd synthetic_data_pipeline
sh generate_positive_negative.sh
```

2. **Train the classifier**:

```bash
cd classifier_training
sh train_classifier.sh
```

### 6. Notes

- Ensure that the environment has all the necessary dependencies installed.
- The generated synthetic data and trained models will be saved in the specified directories.
- Modify the configurations in the shell scripts to adjust model parameters and dataset settings.
- You can fine-tune the model further by adjusting hyperparameters or training for more epochs.

