import pandas as pd

def createGroups(data_file, verbose):
    """
    Data is read as:

        #  Attribute                     Domain
        -- -----------------------------------------
      -- 1. name: distinct for each instance and represented numerically
      -- 2. hobby: nominal values ranging between 1 and 3
      -- 3. age: nominal values ranging between 1 and 4
      -- 4. educational level: nominal values ranging between 1 and 4
      -- 5. marital status: nominal values ranging between 1 and 4
      -- 6. class: nominal value between 1 and 3

     We'll return a set of 5 data partitions stratified by class.

    :param data_file: The name of the file to load the raw machine data
    :param verbose: Used to indicate whether to log detailed information
    """

    # Drop the estimation from the data set - access it separately
    orig_labels = ["name", "hobby", "age", "education", "marital", "classification"]
    rearr_labels = ["classification", "hobby", "age", "education", "marital"]

    # Labels for features which have real values for processing
    range_labels = ["hobby", "age", "education", "marital"]

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
    createGroups("./hayes-roth.data", False)