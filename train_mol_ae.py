# this trains the molecular autoencoder

from mol_encode import MolecularAutoencoder
from os import mkdir
import argparse
import pandas as pd
import tensorflow as tf


##### Runtime variables
input_data_csv = "data/chembl_100k.csv"
latent_dim = 64
max_length = 120
num_epochs = 50
model_save_name = f"ae_onehot_cnn_maxlen{max_length}_latentdim{latent_dim}"


#### execution
gpus = tf.config.list_physical_devices("GPU")
print("Num GPUs Available: ", len(gpus))


if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=6000)]
        )
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


# Initialize and build model
autoencoder = MolecularAutoencoder(
    latent_dim, max_length, log_dir=f"tensorboard_logs/{model_save_name}"
)
autoencoder.build_vocab()  # Use default vocabulary or provide custom
model = autoencoder.build_model()

# Train
history = autoencoder.train(input_data_csv, epochs=num_epochs)

# Save model
mkdir("saved_models")
autoencoder.save_model(f"saved_models/{model_save_name}")

# Sanity check with a single molecule
result = autoencoder.embed_and_decode(
    "COC1=C(C=C(C=C1)C(F)(F)F)N2[C@H](C3=C(C(=CC=C3)F)N=C2N4CCN(CC4)C5=CC(=CC=C5)OC)CC(=O)O"
)
print(result["embedding"])  # numpy array
print(result["decoded"])  # reconstructed SMILES
