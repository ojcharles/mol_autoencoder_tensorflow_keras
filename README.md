# mol_autoencoder_tensorflow_keras
A molecular autoencoder using tensorflow and keras. Written to be relatively simple to follow, aiming for brevity.


Autoencoder framework:
- Code kept relateively simple with tf, keras, np, pandas
- rdkit only used for molecule standardisation
- One-hot representation of molecules using a pre-definded, but user-alterable vocabulary
- Convolutional layers then are applied to this sparse matrix until we have a latent space
- The model can be saved
- For a given set of new inputs the model will return the decoded smiles as well as the latent representation


Future plans
- implement other frameworks such as graph inputs, fully connected layers, and attention


Oscar Charles 2024