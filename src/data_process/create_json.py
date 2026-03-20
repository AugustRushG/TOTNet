import os
import json

def create_image_json(directory, output_file, base_path="/data/local-files?d=tta_dataset/images/"):
    """
    Create a JSON file listing all images in a directory.

    Args:
        directory (str): Path to the directory containing images.
        output_file (str): Path to save the generated JSON file.
        base_path (str): Base path to prepend to each image path in the JSON.
    """
    # Initialize the list for JSON entries
    json_data = []

    # Walk through the directory and find all image files
    for root, _, files in os.walk(directory):
        for idx, file_name in enumerate(sorted(files), start=1):
            # Check if the file is an image
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                # Construct the relative path for the image
                relative_path = os.path.relpath(os.path.join(root, file_name), directory)
                full_path = os.path.join(base_path, relative_path).replace("\\", "/")  # Append to base_path only

                # Append the JSON entry
                json_data.append({
                    "id": idx,
                    "data": {
                        "img": full_path
                    }
                })

    # Save the JSON data to the output file
    with open(output_file, "w") as json_file:
        json.dump(json_data, json_file, indent=2)
    print(f"JSON file created: {output_file}")

# Example usage
directory = "/home/s224705071/github/PhysicsInformedDeformableAttentionNetwork/data/tta_dataset/training/images/24Paralympics_FRA_M9_Ma_AUS_v_Didier_FRA/Game_1"  # Replace with the actual directory
output_file = "image_list.json"  # Path to save the JSON
base_path = '/data/local-files?d=tta_dataset/training/images/24Paralympics_FRA_M9_Ma_AUS_v_Didier_FRA/Game_1'
create_image_json(directory, output_file, base_path)
