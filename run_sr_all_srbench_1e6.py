import subprocess
import evaluation
import json
import os
from utils import load_srbench_dataset

dataset_list = [
    "feynman_I_6_2", "feynman_I_6_2a", "feynman_I_6_2b", "feynman_I_8_14", "feynman_I_9_18",
    "feynman_I_10_7", "feynman_I_11_19", "feynman_I_12_1", "feynman_I_12_2", "feynman_I_12_4",
    "feynman_I_12_5", "feynman_I_12_11", "feynman_I_13_4", "feynman_I_13_12", "feynman_I_14_3",
    "feynman_I_14_4", "feynman_I_15_3t", "feynman_I_15_3x", "feynman_I_15_10", "feynman_I_16_6",
    "feynman_I_18_4", "feynman_I_18_12", "feynman_I_18_14", "feynman_I_24_6", "feynman_I_25_13",
    "feynman_I_26_2", "feynman_I_27_6", "feynman_I_29_4", "feynman_I_29_16", "feynman_I_30_3",
    "feynman_I_30_5", "feynman_I_32_5", "feynman_I_32_17", "feynman_I_34_1", "feynman_I_34_8",
    "feynman_I_34_14", "feynman_I_34_27", "feynman_I_37_4", "feynman_I_38_12", "feynman_I_39_1",
    "feynman_I_39_22", "feynman_I_40_1", "feynman_I_41_16", "feynman_I_43_16",
    "feynman_I_43_31", "feynman_I_44_4", "feynman_I_48_2",
    "feynman_I_50_26", "feynman_II_2_42", "feynman_II_3_24", "feynman_II_4_23", "feynman_II_6_11",
    "feynman_II_6_15a", "feynman_II_6_15b", "feynman_II_8_7", "feynman_II_8_31", "feynman_II_10_9",
    "feynman_II_11_3", "feynman_II_11_20", "feynman_II_11_27", "feynman_II_11_28",
    "feynman_II_13_23", "feynman_II_13_34", "feynman_II_15_4", "feynman_II_15_5", "feynman_II_21_32",
    "feynman_II_24_17", "feynman_II_27_16", "feynman_II_27_18", "feynman_II_34_2", "feynman_II_34_2a",
    "feynman_II_34_11", "feynman_II_34_29a", "feynman_II_34_29b", "feynman_II_35_18", "feynman_II_35_21",
    "feynman_II_36_38", "feynman_II_37_1", "feynman_II_38_3", "feynman_II_38_14", "feynman_III_4_32",
    "feynman_III_4_33", "feynman_III_7_38", "feynman_III_8_54", "feynman_III_9_52", "feynman_III_10_19",
    "feynman_III_12_43", "feynman_III_13_18", "feynman_III_14_14", "feynman_III_15_12", "feynman_III_15_14",
    "feynman_III_15_27", "feynman_III_19_51", "feynman_III_21_20", "feynman_test_1",
    "feynman_test_2", "feynman_test_3", "feynman_test_4", "feynman_test_5", "feynman_test_6",
    "feynman_test_7", "feynman_test_8", "feynman_test_9", "feynman_test_11",
    "feynman_test_12", "feynman_test_13", "feynman_test_14", "feynman_test_15", "feynman_test_16",
    "feynman_test_17", "feynman_test_18", "feynman_test_19", "strogatz_bacres1",
    "strogatz_bacres2", "strogatz_barmag1", "strogatz_barmag2", "strogatz_glider1", "strogatz_glider2",
    "strogatz_lv1", "strogatz_lv2", "strogatz_predprey1", "strogatz_predprey2", "strogatz_shearflow1",
    "strogatz_shearflow2", "strogatz_vdp1", "strogatz_vdp2", "feynman_I_39_11", "feynman_I_43_43", "feynman_I_47_23", 
    "feynman_III_17_37", "feynman_test_20", "feynman_II_13_17"
]

quick_dataset_list = {
    "feynman_I_39_11", "feynman_I_43_43"
    #, "feynman_I_47_23", "feynman_III_17_37", "feynman_test_20", "feynman_II_13_17"
    #"feynman_I_6_2": "exp(-(theta/sigma)**2/2)/(sqrt(2*pi)*sigma)",
    #"feynman_test_5": "2*pi*d**(3/2)/sqrt(G*(m1+m2))",
    #"feynman_III_15_27": "2*pi*alpha/(n*d)",
    #"strogatz_predprey2": "y * ( (x)/(1+x) - 0.075 * y )",
    #"strogatz_glider1": "-0.05 * x**2 - sin(y)"
                      }

max_samples = 1000
seed = 42
results_dir = "results_pysr"

accurate_predictions = 0
total_predictions = 0
total_time = 0
total_mse = 0
total_r2 = 0

for dataset in dataset_list:

    _, _, truth = load_srbench_dataset(dataset)

    if "=" in truth:
        truth = truth.split("=")[1].strip()
        print(truth)

    # Define the command as a list of arguments
    command = [
        "python", 
        "run_pysr_srbench.py", 
        "--dataset=" + dataset, 
        "--max_evals=1000000",
    ]

    # Execute the command
    result = subprocess.run(command, capture_output=True, text=True)

    # Check the output or errors
    print("Output:", result.stdout)
    #if result.stderr:
        #print("Errors:", result.stderr)

    # Construct the path exactly how the script does
    json_path = os.path.join(results_dir, f"{dataset}_n{max_samples}_seed{seed}.json")

    with open(json_path, 'r') as f:
        results = json.load(f)
        predicted_expression_str = results['best_equation']
        fit_time = results['fit_time_seconds']
        r2_value = results['test_r2']
        mse_value = results['test_mse']

    print("dataset: " + dataset + ", prediction: " + predicted_expression_str)

    check_result = evaluation.check_symbolic_match(predicted_expression_str, evaluation.parse_ground_truth(truth))
    print(check_result)

    if (check_result['match']):
        accurate_predictions += 1
    total_predictions += 1

    total_time += fit_time
    total_r2 += r2_value
    total_mse += mse_value

    print("total_predictions=" + str(total_predictions))
    print("accurate_predictions=" + str(accurate_predictions))
    print("Accuracy=" + str(accurate_predictions / total_predictions))
    print("Total time=" + str(total_time))
    print("Average time per dataset=" + str(total_time / total_predictions))
    print("Average MSE=" + str(total_mse / total_predictions))
    print("Average R^2=" + str(total_r2 / total_predictions))
    print("\n")