import pandas as pd

def createGroups(data_file, verbose):
    """
    Read the glass data into a pandas dataframe with the data we care about.
    Data is read as:

        #  Attribute                     Domain
                -- -----------------------------------------
        1. Id number: 1 to 214
        2. RI: refractive index
        3. Na: Sodium (unit measurement: weight percent in corresponding oxide, as 
                        are attributes 4-10)
        4. Mg: Magnesium
        5. Al: Aluminum
        6. Si: Silicon
        7. K: Potassium
        8. Ca: Calcium
        9. Ba: Barium
        10. Fe: Iron
        11. Type of glass: (class attribute)
            -- 1 building_windows_float_processed
            -- 2 building_windows_non_float_processed
            -- 3 vehicle_windows_float_processed
            -- 4 vehicle_windows_non_float_processed (none in this database)
            -- 5 containers
            -- 6 tableware
            -- 7 headlamps

     We'll return a set of 5 data partitions stratified by class.

    :param data_file: The name of the file to load the raw machine data
    :param verbose: Used to indicate whether to log detailed information
    """

    # Drop the estimation from the data set - access it separately
    orig_labels = ["id", "ri", "na", "mg", "al", "si", "k", "ca", "ba", "fe", "classification"]
    rearr_labels = ["classification", "ri", "na", "mg", "al", "si", "k", "ca", "ba", "fe"]

    # Labels for features which have real values for processing
    range_labels = ["ri", "na", "mg", "al", "si", "k", "ca", "ba", "fe"]

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
    createGroups("./glass.data", False)