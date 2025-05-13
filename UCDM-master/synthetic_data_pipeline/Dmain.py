import os
import time
import logging
import tempfile
try:
    from .dataset import *
    from .generator import *
except ImportError:
    import sys, os
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))
    from dataset import *
    from generator import *

import torch.multiprocessing as mp
from argparse import ArgumentParser

import tempfile
import torch.distributed as dist

def initialize_distributed(rank, world_size, args, temp_dir):
    """
    Initializes the distributed training environment, supporting both rigid and elastic launch modes.
    """
    if args.distributed:
        assert dist.is_available() and torch.cuda.is_available()

        if args.rigid_launch:
            logging.info("Rigid launch for distributed training.")
            assert temp_dir, "Temporary directory cannot be empty!"
            init_method = f"file://{os.path.join(os.path.abspath(temp_dir), '.torch_distributed_init')}"
            dist.init_process_group("nccl", init_method=init_method, rank=rank, world_size=world_size)
            local_rank = rank
            os.environ["WORLD_SIZE"] = str(world_size)
            os.environ["LOCAL_RANK"] = str(rank)
        else:
            logging.info("Elastic launch for distributed training.")
            world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
            rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
            dist.init_process_group("nccl", init_method="env://", world_size=world_size, rank=rank)
            local_rank = int(os.environ.get("LOCAL_RANK", "0")) or rank % world_size
            os.environ["WORLD_SIZE"] = os.environ.get("WORLD_SIZE", str(world_size))

        torch.cuda.set_device(local_rank)
        train_device = torch.device(f"cuda:{local_rank}")
        logging.info(f"Distributed setup complete with {world_size} GPUs.")
        return train_device, local_rank, world_size
    else:
        rank = local_rank = 0
        torch.cuda.set_device(local_rank)
        train_device = torch.device(f"cuda:{local_rank}")
        logging.info("Running on a single GPU (non-distributed).")
        return train_device

def train(rank=0, args=None, temp_dir="", train_set=None, pair_dataset=None):
    """
    Main training function that supports distributed training, data generation, and pseudo-label updates.
    """
    # Initialize distributed training setup
    if args.distributed:
        train_device, local_rank, world_size = initialize_distributed(rank, args.num_gpus, args, temp_dir)
    else:
        train_device = initialize_distributed(rank, args.num_gpus, args, temp_dir)
    
    # Load the generator model
    generator = LoadGenerator(args.class_prompt, local_rank, batch_num=args.batch_num, model_id=args.model_path)
    logging.info("Diffusion model loaded.")
    pair_dataset.generator = generator

    logging.info("Generating data for the entire dataset.")
    unlabeled_dataloader, unlabeled_indices = process_data_distributed(train_set, args.known_class, args.unknown_class, args.new_class, world_size=world_size, rank=rank)
    
    # Data generation and saving
    for batch_idx, batch_data in enumerate(tqdm(unlabeled_dataloader, desc="Generate Data")):
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

        u_inputs, u_labels_, u_index = batch_data
        with torch.no_grad():
            pair_dataset.generate_and_store_data(u_index, u_inputs)

        # Save generated data every 20 batches
        if batch_idx != 0 and batch_idx % 20 == 0:
            pair_dataset.save_generated_data()
            logging.info(f"Batch {batch_idx} saved.")

    # Set positive-negative pairs and save the generated data
    pair_dataset.update_positive_negative_pairs()
    pair_dataset.save_generated_data()
    logging.info(f"Total collected indices: {len(pair_dataset.created_indices)}")




def main():
    # Set PyTorch CUDA configuration
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # Create ArgumentParser object
    parser = ArgumentParser()

    # Distributed training settings
    parser.add_argument("--distributed", default=True, action="store_true", help="Whether to use distributed training")
    parser.add_argument("--gpu", default="2,3,4,5", help="GPUs used for distributed training")
    parser.add_argument("--rigid_launch", default=True, action="store_true", help="Whether to use torch multiprocessing spawn")
    parser.add_argument("--num_gpus", default=4, type=int, help="Number of GPUs for distributed training")
    parser.add_argument("--train_device", default="cuda:0", type=str, help="Device to use for training")

    # Data-related settings
    parser.add_argument("--class_prompt", default='"airplane", "automobile"', type=str, help="Training categories (str)")
    parser.add_argument("--data_dir", default='/data', type=str, help="Directory for data")
    parser.add_argument("--save_path", default='/data', type=str, help="Path to save results")
    parser.add_argument("--data_name", default='cifar10', type=str, help="Name of the dataset")
    parser.add_argument("--batch_size", default=20, type=int, help="batch size")
    parser.add_argument("--batch_num", default=5, type=int, help="The number of samples processed by each gpu")

    # model setting
    parser.add_argument("--model_path", default='/data', type=str, help="Path to load diffusion model")


    # Class settings
    parser.add_argument("--known_class", default='0,1', type=str, help="Target classes (comma-separated)")
    parser.add_argument("--unknown_class", default='2,3,4,5,6,7', type=str, help="Unknown categories in unlabeled data (comma-separated)")
    parser.add_argument("--new_class", default='8,9', type=str, help="New categories in test data (comma-separated)")

    # Test size settings
    parser.add_argument("--test_size", default='2000,2000,2000', type=str, help="Sizes for known, unknown, and new classes (comma-separated)")



    # Argument parsing and pre-processing
    args = parser.parse_args()

    # Convert input arguments to appropriate types
    args.known_class = list(map(int, args.known_class.split(',')))
    args.unknown_class = list(map(int, args.unknown_class.split(',')))
    args.new_class = list(map(int, args.new_class.split(',')))
    args.test_size = tuple(map(int, args.test_size.split(',')))
    args.class_prompt = [cat.strip().strip('"') for cat in args.class_prompt.split(",")]

    # Initialize logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Arguments provided:")
    for key, value in vars(args).items():
        logging.info(f"{key}: {value}")

    # Set CUDA devices for training
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # Set GPUs to be used



    # Load the dataset
    train_set = CustomImageDataset(known_class = args.known_class, data_path=args.data_dir, dataset_name=args.data_name, download=False, is_train=True)
    logging.info("Dataset loaded.")

    # Timer to measure elapsed time
    start_time = time.time()

    # Shared memory setup for multiprocessing
    manager = mp.Manager()
    shared_dict = manager.dict()
    shared_created_indices = manager.list()
    shared_value = manager.Value('b', False)  # Boolean value for synchronization

    # Initialize pair_dataset
    pair_dataset = PositiveNegativePairDataset(generator=None, known_class=args.known_class, shared_dict=shared_dict, shared_list=shared_created_indices, shared_value=shared_value, save_path=args.save_path)
    logging.info("Shared pair_dataset created.")

    # Process data and prepare indices
    indices = process_data_distributed(train_set, args.known_class, args.unknown_class, args.new_class)
    pair_dataset.set_total_indices(indices)

    # Distributed training
    if args.distributed and args.rigid_launch:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Launch distributed training processes
            mp.spawn(train, args=(args, temp_dir, train_set, pair_dataset), nprocs=args.num_gpus)
            logging.info("Distributed training processes spawned.")


    logging.info("All processes completed.")
    logging.info("Dataset has been saved!")

    # Log elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Elapsed time: {elapsed_time:.4f} seconds")


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()