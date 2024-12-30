import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import datetime
from tensorflow.keras import layers, Model


class MolecularAutoencoder:
    '''
    An autoencoder class used as the base for variant frameworks.
    This specific model:
    - Takes as an input a sequence of integers, where the integer is the vocab index from smiles. i.e. CCO would become [3,3,7] ( plus padding )
    - The network then learns transforms each integer into a learned dense vector of size input_embed_dim
    - At each atom position, convolutional layers then take the preceeding, current, and future atom dense vector, and return a 64-length new vector for each position. 32 on next iteration.
    - Eventually a fully connected layer is met between the encoder and decoded CNN's. it is the neuron values of this vector we take as our latent represenatation of molecules.
    '''

    def __init__(self, latent_dim=64, max_length=128, input_embed_dim=8, cnn_kernel_size=3, log_dir="logs"):
        # Default vocabulary including special tokens and common molecular symbols
        self.default_vocab = ['<PAD>', '<START>', '<END>', 'C', 'c', 'N', 'n', 'O', 'o', 'H', 'S', 's', 'F', 'Cl', 'Br', 'I',
                            'P', 'Si', 'As', 'Se', '(', ')', '[', ']', '=', '#', '+', '-', '1', '2', '3', '4', '5', '6',
                            '7', '8', '9', '0', '@', '/', '\\', '.']
        self.log_dir = log_dir
        self.vocab = None
        self.char2idx = None
        self.idx2char = None
        self.max_length = max_length
        self.latent_dim = latent_dim
        self.input_embed_dim = input_embed_dim
        self.cnn_kernel_size = cnn_kernel_size
        self.encoder = None
        self.decoder = None
        self.autoencoder = None

    def build_vocab(self, custom_vocab=None):
        self.vocab = custom_vocab if custom_vocab else self.default_vocab
        self.char2idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx2char = {idx: char for idx, char in enumerate(self.vocab)}
        return self.vocab

    def tokenize(self, smiles):
        # Tokenize SMILES string considering multi-character tokens
        tokens = []
        i = 0
        while i < len(smiles):
            if i + 1 < len(smiles) and smiles[i : i + 2] in self.vocab:
                tokens.append(smiles[i : i + 2])
                i += 2
            else:
                if smiles[i] in self.vocab:
                    tokens.append(smiles[i])
                i += 1
        return tokens

    def smiles_to_seq(self, smiles):
        tokens = self.tokenize(smiles)
        seq = [self.char2idx["<START>"]]
        seq.extend(self.char2idx[token] for token in tokens if token in self.char2idx)
        seq.append(self.char2idx["<END>"])
        # Pad sequence
        if len(seq) < self.max_length:
            seq.extend([self.char2idx["<PAD>"]] * (self.max_length - len(seq)))
        return seq[: self.max_length]

    def prepare_data(self, csv_path):
        df = pd.read_csv(csv_path)
        sequences = [self.smiles_to_seq(smiles) for smiles in df["smiles"]]
        return np.array(sequences)

    def build_model(self):
        # Ensure GPU is being used
        physical_devices = tf.config.list_physical_devices("CUDA")
        if physical_devices:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

        # Encoder
        encoder_inputs = layers.Input(shape=(self.max_length,))
        embedding = layers.Embedding(len(self.vocab), self.input_embed_dim)(encoder_inputs)             # Shape: (batch, max_length, input_embed_dim)
        conv1 = layers.Conv1D(filters=64, kernel_size=self.cnn_kernel_size, activation="relu", padding="same")(embedding)  # Shape: (batch, max_length, 64)
        conv2 = layers.Conv1D(filters=32, kernel_size=self.cnn_kernel_size, activation="relu", padding="same")(conv1)
        flatten = layers.Flatten()(conv2)
        encoder_outputs = layers.Dense(self.latent_dim)(flatten)
        self.encoder = Model(encoder_inputs, encoder_outputs, name="encoder")

        # Decoder
        latent_inputs = layers.Input(shape=(self.latent_dim,))
        dense = layers.Dense(self.max_length * 32)(latent_inputs)
        reshape = layers.Reshape((self.max_length, 32))(dense)
        conv3 = layers.Conv1D(filters=64, kernel_size=self.cnn_kernel_size, activation="relu", padding="same")(reshape)
        conv4 = layers.Conv1D(filters=self.max_length, kernel_size=self.cnn_kernel_size, activation="relu", padding="same")(conv3)
        decoder_outputs = layers.Dense(len(self.vocab), activation="softmax")(conv4)
        self.decoder = Model(latent_inputs, decoder_outputs, name="decoder")

        # Autoencoder
        autoencoder_inputs = layers.Input(shape=(self.max_length,))
        encoded = self.encoder(autoencoder_inputs)
        decoded = self.decoder(encoded)
        self.autoencoder = Model(autoencoder_inputs, decoded, name="autoencoder")

        # Compile model
        self.autoencoder.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return self.autoencoder

    def train(self, csv_path, epochs=50, batch_size=32, validation_split=0.2):
        data = self.prepare_data(csv_path)
        target_data = np.roll(data, -1, axis=1)

        # Create log directory with timestamp
        log_dir = f"{self.log_dir}/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        callbacks = [
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True,
                write_images=True,
                update_freq="epoch",
                profile_batch=0,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                min_delta=1e-4,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.2, patience=2, min_delta=1e-4
            ),
        ]

        return self.autoencoder.fit(
            data,
            target_data,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
        )

    def encode(self, smiles):
        seq = np.array([self.smiles_to_seq(smiles)])
        return self.encoder.predict(seq)

    def decode(self, latent_vector):
        output_seq = self.decoder.predict(latent_vector)
        # Convert probabilities to indices
        indices = np.argmax(output_seq[0], axis=1)
        # Convert indices to tokens
        tokens = [self.idx2char[idx] for idx in indices]
        # Remove padding and end tokens
        tokens = [t for t in tokens if t not in ["<PAD>", "<START>", "<END>"]]
        return "".join(tokens)

    def get_embedding(self, smiles_string):
        """Get latent space embedding for a single SMILES string"""
        return self.encode(smiles_string)[0]  # Return 1D array instead of 2D

    def embed_and_decode(self, smiles_string):
        """Get both embedding and decoded SMILES for a molecule"""
        embedding = self.get_embedding(smiles_string)
        decoded = self.decode(embedding.reshape(1, -1))
        return {"original": smiles_string, "embedding": embedding, "decoded": decoded}

    def prepare_batch(self, smiles_list):
        """Convert a list of SMILES to padded sequence batch"""
        sequences = [self.smiles_to_seq(smiles) for smiles in smiles_list]
        return np.array(sequences)

    def batch_encode(self, smiles_list, batch_size=32):
        """Encode multiple SMILES strings efficiently"""
        sequences = self.prepare_batch(smiles_list)
        return self.encoder.predict(sequences, batch_size=batch_size)

    def batch_decode(self, latent_vectors, batch_size=32):
        """Decode multiple latent vectors efficiently"""
        output_seqs = self.decoder.predict(latent_vectors, batch_size=batch_size)
        decoded_smiles = []
        for seq in output_seqs:
            indices = np.argmax(seq, axis=1)
            tokens = [self.idx2char[idx] for idx in indices]
            tokens = [t for t in tokens if t not in ["<PAD>", "<START>", "<END>"]]
            decoded_smiles.append("".join(tokens))
        return decoded_smiles

    def process_csv_batch(self, csv_path, batch_size=32):
        """Process entire CSV file in batches"""
        df = pd.read_csv(csv_path)
        embeddings = self.batch_encode(df["smiles"].tolist(), batch_size=batch_size)
        decoded = self.batch_decode(embeddings, batch_size=batch_size)
        return {
            "embeddings": embeddings,
            "decoded": decoded,
            "original": df["smiles"].tolist(),
        }

    def save_model(self, path):
        # Save with compile=True to preserve optimizer state
        self.autoencoder.save(
            f"{path}_autoencoder.h5", save_format="h5", include_optimizer=True
        )
        self.encoder.save(
            f"{path}_encoder.h5", save_format="h5", include_optimizer=True
        )
        self.decoder.save(
            f"{path}_decoder.h5", save_format="h5", include_optimizer=True
        )

        # Save model configuration
        config = {
            "vocab": self.vocab,
            "char2idx": self.char2idx,
            "idx2char": self.idx2char,
            "max_length": self.max_length,
            "latent_dim": self.latent_dim,
        }
        with open(f"{path}_config.pkl", "wb") as f:
            pickle.dump(config, f)

    def load_model(self, path):
        # Load configuration first
        with open(f"{path}_config.pkl", "rb") as f:
            config = pickle.load(f)
        self.__dict__.update(config)

        # Load models with custom_objects if needed
        self.autoencoder = tf.keras.models.load_model(
            f"{path}_autoencoder.h5", compile=True
        )
        self.encoder = tf.keras.models.load_model(f"{path}_encoder.h5", compile=True)
        self.decoder = tf.keras.models.load_model(f"{path}_decoder.h5", compile=True)
