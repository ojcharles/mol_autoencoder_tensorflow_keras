# a script to run everything

##### code environment
conda create -n deep_env -c conda-forge python=3.9 tensorflow-gpu=2.10 keras=2.10 pandas numpy scikit-learn
conda activate deep_env
# conda env export | grep -v "^prefix: " > environment.yml

# alternatively use the environment.yml file


##### data pre-processing
# download all small molecules from chembl ~1.9m - manual step
# now only pipe the smiles into a file
awk -F "\"*;\"*" '{print $32}' data/chembl_part1.csv > data/chembl_smiles_raw.csv
awk -F "\"*;\"*" '{print $32}' data/chembl_part2.csv >> data/chembl_smiles_raw.csv

# the data isn't totally tidy
# remove empirical forumlae
sed -i '/H[1-9]\{2\}/d' data/chembl_smiles_raw.csv
# remove very small molecules
sed -i '/^[a-z]{\1,\2}$/d' data/chembl_smiles_raw.csv

# normalise and ensure validity of all smiles
python3 mol_encode/smiles_pre_processing.py -i data/chembl_smiles_raw.csv -c data/chembl_corrupt.csv -p data/chembl_2m.csv -t 10

# make subsets
head -100000 data/chembl_2m.csv > data/chembl_100k.csv
head -500000 data/chembl_2m.csv > data/chembl_500k.csv
# a quick set for assessing model inference in the notebook
echo "smiles" > data/new_mols.csv ; tail -100 data/chembl_2m.csv >> new_mols.csv


##### training model
python3 train_mol_ae.py
# to monitor in tensorboard run in terminal: ` tensorboard --logdir tensorboard_logs `