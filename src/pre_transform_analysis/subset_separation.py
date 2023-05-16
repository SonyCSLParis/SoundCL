import os

root1="../../dataset/SpeechCommands/FemaleCommands/"
root2="../../dataset/SpeechCommands/MaleCommands/"

# Path to the validation_list.txt file
validation_file = "validation_list.txt"

# Path to the testing_list.txt file
testing_file = "testing_list.txt"

tasks=[[root1+validation_file,root1+testing_file],[root2+validation_file,root2+testing_file]]

for files_to_treat in tasks:
    for dataset_file in files_to_treat:
        # Read the .txt file and store the file paths
        with open(dataset_file, "r") as file:
            file_paths = file.readlines()
        file_paths = [path.strip() for path in file_paths]

        # List to store the valid file paths
        valid_file_paths = []

        # Loop through the file paths and check if the files exist
        for file_path in file_paths:
            if os.path.exists(root2+file_path):
                valid_file_paths.append(file_path)
            else:
                print(f"File not found: {file_path}")

        # Update the .txt file with the valid file paths
        with open(dataset_file, "w") as file:
            file.write("\n".join(valid_file_paths))

        print(dataset_file+" file updated successfully.")