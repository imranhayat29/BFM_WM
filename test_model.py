'''
Script to test wall model on different cases interactively

Models should be saved in ./models/

Example:
    python test_model.py wall_model.pth 0 # Not save the plots
    python test_model.py wall_model.pth 1 # Save the plots
'''
import sys
import os
from src.wall_model import WallModel
from wall_model_cases import TURB_CASES, TURB_CASES_TREE, print_dataset_tree
import Levenshtein # This is to find similar dataset names

# --- Read in command line arguments ---
# Using argparse here for flexibility and future expansion
import argparse
parser = argparse.ArgumentParser(
                    prog='BFM Tester',
                    description='Test wall model on different datasets and compare results.',
                    epilog='Text at the bottom of help')

parser.add_argument('--model_path', type=str, nargs='?', default=None, 
             help='Path to the wall model file.')
parser.add_argument('--save_mode', type=int, nargs='?', default=0,
             help='Set to 1 to save plots, 0 otherwise.')
parser.add_argument('--mask_threshold', type=float, nargs='?', default=None,
             help='Threshold for splitting low values in the results.')
parser.add_argument('--mask_threshold_Re', type=float, nargs='?', default=None,
             help='Threshold for splitting low Re values in the results.')
parser.add_argument('--save_results', type=int, nargs='?', default=1,
             help='Set to 1 to save results, 0 otherwise.')

args = parser.parse_args()
model_path = args.model_path
save_mode = bool(args.save_mode)
mask_threshold = args.mask_threshold
mask_threshold_Re = args.mask_threshold_Re
save_results = bool(args.save_results)

# BFM_WM_v2: create the output folders (they are not in the repository); upstream assumed they existed
if save_mode:
    os.makedirs('./results', exist_ok=True)
if save_results:
    os.makedirs('./paper_plots/results_for_plotting_spatial', exist_ok=True)

if mask_threshold_Re is not None and mask_threshold is not None:
    print("Warning: as mask_threshold_Re is provided, mask_threshold will be ignored.")
    mask_threshold = None

##########################################################

def get_closest_matches(invalid_key, valid_keys, num_suggestions=10):
    """
    Finds the closest matching keys from a list of valid keys
    based on Levenshtein distance.
    """
    distances = []
    for key in valid_keys:
        distance = Levenshtein.distance(invalid_key, key)
        distances.append((distance, key))

    # Sort by distance and return the top N
    distances.sort()
    return [key for distance, key in distances[:num_suggestions]]

def load_and_test_model(model_path, test_dataset=None, wall_model=None):

    if wall_model is None:
        wall_model = WallModel.load_compact(model_path, device="cpu")

    # Test each dataset
    print(f"\nTesting on {test_dataset}...")

    save_path = "./results/" if save_mode else None

    results = wall_model.test_external_dataset(
        dataset_key=test_dataset,
        tauw=True,
        mask_threshold=mask_threshold,
        mask_threshold_Re=mask_threshold_Re,
        save_path=save_path,
        LogTransform=wall_model.config.get('training', {}).get('LogTransform', False),
        near_wall=wall_model.config.get('data', {}).get('near_wall_threshold', -1.0),
        SaveResults=save_results
    )

    return results, wall_model


# Outer code handling dataset input
valid_datasets = TURB_CASES

model_path = os.path.join('./models', model_path) if model_path else None

test_dataset = None
wall_model   = None

while True:
    test_dataset = input(f"Enter the dataset to test (P for printing all; R for regression plot): ")
    
    if test_dataset == 'P':
# Call the function to print the dataset tree
        print_dataset_tree(TURB_CASES_TREE)
        continue
    elif test_dataset == 'R':
        # Do regression plot without saving the plots
        if wall_model is None:
            results, wall_model = load_and_test_model(model_path, 'SYN')
        wall_model.load_data()
        wall_model.test(plot=True, plot_frequency=True)
        continue
    elif test_dataset == 'Rs':
        # Do regression plot but save the plots
        if wall_model is None:
            results, wall_model = load_and_test_model(model_path, 'SYN')
        wall_model.load_data()
        os.makedirs('./paper_plots/regression', exist_ok=True)  # BFM_WM_v2: not in the repository
        wall_model.test(plot=False,plot_frequency=True, save_path="./paper_plots/regression/")
        continue
    elif test_dataset == 'Q':
        print("Exiting the program.")
        break
    elif test_dataset not in valid_datasets:
        print(f"Invalid dataset '{test_dataset}'.")
        closest_matches = get_closest_matches(test_dataset, valid_datasets)
        print("Did you mean one of these?")
        for match in closest_matches:
            print(f"- {match}")
        continue # Let the user try againe
    else:
        # Valid dataset entered, test the model
        if wall_model is None:
            results, wall_model = load_and_test_model(model_path, test_dataset)
        else:
            results, _ = load_and_test_model(model_path, test_dataset, wall_model=wall_model)

        # if results is not None:
        #     wall_model.test_external_dataset(
        #         dataset_key=test_dataset,
        #         purpose = 1
        #     )

        print(results)
