import pandas as pd

def createGroups(data_file, verbose):
    """
    Attribute Information:
	1. Class Name: 3 (L, B, R)
	2. Left-Weight: 5 (1, 2, 3, 4, 5)
	3. Left-Distance: 5 (1, 2, 3, 4, 5)
	4. Right-Weight: 5 (1, 2, 3, 4, 5)
	5. Right-Distance: 5 (1, 2, 3, 4, 5)

    :param data_file: The name of the file to load the raw machine data
    :param verbose: Used to indicate whether to log detailed information
    """

    # Drop the estimation from the data set - access it separately
    # Drop model and vendor name as identifying info
    orig_labels = ["classification", "left-weight", "left-distance", "right-weight", "right-distance"]
    rearr_labels = orig_labels

    # Labels for features which have real values for processing
    range_labels = ["left-weight", "left-distance", "right-weight", "right-distance"]

    # Uses comma as delimiter - ignore present header row
    data = pd.read_csv(data_file, sep=",", header=None, names=orig_labels)

    # Move classification to the front
    data = data[rearr_labels]

    # Delete rows with unknown values
    data = data.mask(data.eq('?')).dropna() 
    if verbose: print("Normalizing data for labels: {}".format(range_labels))

    for label in range_labels:

        # Convert back to numbers
        data[label] = [float(v) for v in data[label]]
        
        # Normalize the range
        min_val = min(data[label])
        max_val = max(data[label])
        data[label] = [v - min_val for v in data[label]]
        if min_val != max_val:
            data[label] = [v / (max_val - min_val) for v in data[label]]

    with open("data.json", "w") as f:
        f.write("{}".format(data.to_json()))

if __name__ == "__main__":
    createGroups("./balance-scale.data", False)