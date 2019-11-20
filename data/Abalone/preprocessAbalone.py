import pandas as pd

def createGroups(data_file, verbose):
    """
    Read the Abalone data into a pandas dataframe with the data we care about.
    Data is read as:

        Name		Data Type	Meas.	Description
        ----		---------	-----	-----------
        Sex		nominal			M, F, and I (infant)
        Length		continuous	mm	Longest shell measurement
        Diameter	continuous	mm	perpendicular to length
        Height		continuous	mm	with meat in shell
        Whole weight	continuous	grams	whole abalone
        Shucked weight	continuous	grams	weight of meat
        Viscera weight	continuous	grams	gut weight (after bleeding)
        Shell weight	continuous	grams	after being dried
        Rings		integer			+1.5 gives the age in years

     We'll return a set of 5 data partitions stratified by class.

    :param data_file: The name of the file to load the raw machine data
    :param verbose: Used to indicate whether to log detailed information
    """

    # Drop the estimation from the data set - access it separately
    # Drop model and vendor name as identifying info
    orig_labels = ["sex", "length", "diameter", "height", "whole weight", "shucked weight", "viscera weight", "shell weight", "classification"]
    rearr_labels = ["classification", "sex", "length", "diameter", "height", "whole weight", "shucked weight", "viscera weight", "shell weight"]

    # Labels for features which have real values for processing
    range_labels = ["length", "diameter", "height", "whole weight", "shucked weight", "viscera weight", "shell weight"]

    # Uses comma as delimiter - ignore present header row
    data = pd.read_csv(data_file, sep=",", header=None, names=orig_labels)

    # Move classification to the front
    data = data[rearr_labels]

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
    createGroups("./abalone.data", False)