
# From https://github.com/lucianistrati/random_seed_setter repo
# This repository contains a way to instantiate a random seed across multiple python libraries

import random


def reset_numpy_seed(seed_value: int = 6174):
    try:
        # Set NumPy random seed
        import numpy as np
        np.random.seed(seed_value)
        print(f'NumPy random seed set with value: {seed_value}')
    except Exception as e:
        print(f'NumPy random seed was not set: {e}')
    return


def reset_tensorflow_seed(seed_value: int = 6174):
    try:
        # Set TensorFlow random seed
        import tensorflow as tf
        success = False
        # Here we have 2 different ways to set the seed
        # depending on the version of TensorFlow
        try:
            tf.random.set_seed(seed_value)
            success = True
        except Exception as e:
            pass
        try:
            tf.set_random_seed(seed_value)
            success = True
        except Exception as e:
            pass
        if success:
            print(f'TensorFlow random seed set with value: {seed_value}')
        else:
            print(f'TensorFlow random seed was not set')
    except Exception as e:
        print(f'TensorFlow random seed was not set: {e}')
    return


def reset_jax_seed(seed_value: int = 6174):
    try:
        # Set Jax random seed
        import jax
        jax.random.PRNGKey(seed)
        print(f'Jax random seed set with value: {seed_value}')
    except Exception as e:
        print(f'Jax random seed was not set: {e}')
    return


def reset_torch_seed(seed_value: int = 6174):
    try:
        # Set PyTorch random seed
        import torch
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)  # if you are using multiple GPUs
        print(f'PyTorch random seed set with value: {seed_value}')
    except Exception as e:
        print(f'PyTorch random seed was not set: {e}')
  

def set_random_seeds(seed_value: int = 6174):
    # Set Python random seed
    random.seed(seed_value)
    reset_numpy_seed(seed_value)
    reset_tensorflow_seed(seed_value)
    reset_torch_seed(seed_value)
    reset_jax_seed(seed_value)


def main():
    # Set the desired seed value here
    seed = 6174
    # Most people just go with 42, but I actually like more 6174 because it is
    # Kaprekar's constant, but each with their own
    set_random_seeds(seed) # Set random seeds across: numpy, random, tensorflow, pytorch and jax libraries 


if __name__ == '__main__':
    main()
