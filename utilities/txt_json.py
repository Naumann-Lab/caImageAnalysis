import json
import numpy as np
import ast


def convert_numpy(obj):
    """Recursively convert numpy arrays in the dictionary to lists."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to list
    elif isinstance(obj, dict):
        return {key: convert_numpy(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    else:
        return obj


def load_and_convert_to_json(input_file_path, output_file_path):
    """Load a text file containing a Python dictionary, convert it, and save as JSON."""
    try:
        # Read the content of the text file
        with open(input_file_path, 'r') as file:
            content = file.read()

        #get rid of the numpy array syntax
        content = content.replace('array(', '').replace(')', '')
        # Safely evaluate the string to a Python dictionary (assumes the input is a dictionary formatted as a string)
        data_dict = ast.literal_eval(content)

        # Convert numpy arrays to lists
        data_dict = convert_numpy(data_dict)

        # Convert the dictionary to JSON and write to a file
        with open(output_file_path, 'w') as json_file:
            json.dump({'summary_command': data_dict}, json_file, indent=4)

        print(f"Data successfully converted and saved to {output_file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")