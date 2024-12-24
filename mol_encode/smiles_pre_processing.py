#A Python script to read a CSV file containing SMILES strings, identify corrupted SMILES, normalizes valid SMILES, removes salts, and outputs the results to two separate CSV files
# Usage: python3 smiles_csv_parser.py -i input.csv -c corrupted.csv -p processed.csv -t 10

import csv
import argparse
from rdkit import Chem
from rdkit.Chem import SaltRemover
from rdkit.Chem.MolStandardize import Standardizer
from multiprocessing import Pool

# Initialize SaltRemover
remover = SaltRemover.SaltRemover()

# Initialize MolStandardizer
standardizer = Standardizer()


def process_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Corrupted SMILES
        return (smiles, None)
    else:
        # Valid SMILES - remove salts and standardise
        mol = remover.StripMol(mol)
        try:
            mol = standardizer.standardize(mol)
        except Exception as e:
            print(f"Standardization failed: {e}")
            return (smiles, None)
        processed_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        return (smiles, processed_smiles)


def process_smiles(input_file, corrupted_file, processed_file, num_threads):
    with open(input_file, "r") as infile:
        reader = csv.reader(infile)
        next(reader, None)  # Skip header
        smiles_list = [row[0] for row in reader if row]

    with Pool(num_threads) as pool:
        results = pool.map(process_molecule, smiles_list)

    with open(corrupted_file, "w", newline="") as corr_file, open(
        processed_file, "w", newline=""
    ) as proc_file:
        corr_writer = csv.writer(corr_file)
        proc_writer = csv.writer(proc_file)

        corr_writer.writerow(["smiles"])
        proc_writer.writerow(["smiles"])

        for original, processed in results:
            if processed is None:
                corr_writer.writerow([original])
            else:
                proc_writer.writerow([processed])


if __name__ == "__main__":
    # cli argument parsing
    parser = argparse.ArgumentParser(
        description="Process SMILES strings from a CSV file"
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Input CSV file containing SMILES strings"
    )
    parser.add_argument(
        "-c", "--corrupted", required=True, help="Output CSV file for corrupted SMILES"
    )
    parser.add_argument(
        "-p", "--processed", required=True, help="Output CSV file for processed SMILES"
    )
    parser.add_argument(
        "-t", "--threads", required=False, default=2, type=int, help="Number of threads"
    )

    args = parser.parse_args()
    # Use the command line arguments
    process_smiles(args.input, args.corrupted, args.processed, args.threads)
    print(
        f"Processing complete. Check '{args.corrupted}' and '{args.processed}' for results."
    )
