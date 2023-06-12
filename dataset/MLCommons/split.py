import os
import random

data_dir = './subset2/'

classes = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]

validation_split = 0.1
test_split = 0.1

validation_file = data_dir+'validation_list.txt'
test_file = data_dir+'testing_list.txt'

validation_samples = []
test_samples = []

for class_folder in classes:
    class_path = os.path.join(data_dir, class_folder)
    samples = [os.path.join(class_path, file) for file in os.listdir(class_path) if file.endswith('.wav')]
    num_samples = len(samples)
    num_validation = int(num_samples * validation_split)
    num_test = int(num_samples * test_split)

    # Randomly select validation samples
    validation_samples.extend(random.sample(samples, num_validation))

    # Remove validation samples from the list
    samples = list(set(samples) - set(validation_samples))

    # Randomly select test samples
    test_samples.extend(random.sample(samples, num_test))

with open(validation_file, 'w') as f:
    for sample in validation_samples:
        path=os.path.join(os.path.split(os.path.split(sample)[0])[1],os.path.split(sample)[1])
        f.write(path + '\n')

with open(test_file, 'w') as f:
    for sample in test_samples:
        path=os.path.join(os.path.split(os.path.split(sample)[0])[1],os.path.split(sample)[1])
        f.write(path + '\n')

print(f"Validation samples written to: {validation_file}")
print(f"Test samples written to: {test_file}")
