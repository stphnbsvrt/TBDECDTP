import pandas as pd

def createGroups(data_file, verbose):
    """
    Read the Breast Cancer data into a pandas dataframe with the data we care about.
    Data is read as:

        #  Attribute                     Domain
        -- -----------------------------------------
        1. Sample code number            id number
        2. Clump Thickness               1 - 10
        3. Uniformity of Cell Size       1 - 10
        4. Uniformity of Cell Shape      1 - 10
        5. Marginal Adhesion             1 - 10
        6. Single Epithelial Cell Size   1 - 10
        7. Bare Nuclei                   1 - 10
        8. Bland Chromatin               1 - 10
        9. Normal Nucleoli               1 - 10
        10. Mitoses                       1 - 10
        11. Class:                        (2 for benign, 4 for malignant)

     We'll return a set of 5 data partitions stratified by class.

    :param data_file: The name of the file to load the raw machine data
    :param verbose: Used to indicate whether to log detailed information
    """

    # Drop the estimation from the data set - access it separately
    # Drop model and vendor name as identifying info
    orig_labels = ["id", "clump thickness", "size uniformity", "shape uniformity", "marginal adhesion", "epithelial size", "bare nuclei", "chromatin", "normal nucleoli", "mitoses", "classification"]
    rearr_labels = ["classification", "clump thickness", "size uniformity", "shape uniformity", "marginal adhesion", "epithelial size", "bare nuclei", "chromatin", "normal nucleoli", "mitoses"]

    # Labels for features which have real values for processing
    range_labels = ["clump thickness", "size uniformity", "shape uniformity", "marginal adhesion", "epithelial size", "bare nuclei", "chromatin", "normal nucleoli", "mitoses"]

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
    createGroups("./breast-cancer-wisconsin.data", False)