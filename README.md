# mol_autoencoder_tensorflow_keras
A molecular autoencoder using tensorflow and keras. Written to be relatively simple to follow, aiming for brevity.


Autoencoder framework:
- Code kept relatively simple with tf, keras, np, pandas
- rdkit only used for molecule standardisation
- Sequence of integers is the input encoding defined by a defined vocabulary
- Typical character embedding is learned, which represents each character as a n-bit vector
- Convolutional layers then are applied to this dense char_embedding * max_length layer until we have a learned latent space
- The model can be saved
- For a given set of new inputs the model will return the decoded smiles as well as the latent representation


Future plans
- implement other frameworks such as graph inputs, fully connected layers, and attention



Handy references
- Explanantion of 1d convolutions and how the data will look with sequence based encodings https://stackoverflow.com/questions/52352522/how-does-keras-1d-convolution-layer-work-with-word-embeddings-text-classificat

Oscar Charles 2024