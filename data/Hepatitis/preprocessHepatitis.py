import pandas as pd

def createGroups(data_file, verbose):
    """
    Read the Hepatitis data into a pandas dataframe with the data we care about.
    Data is read as:
     
     1. Class: DIE, LIVE
     2. AGE: 10, 20, 30, 40, 50, 60, 70, 80
     3. SEX: male, female
     4. STEROID: no, yes
     5. ANTIVIRALS: no, yes
     6. FATIGUE: no, yes
     7. MALAISE: no, yes
     8. ANOREXIA: no, yes
     9. LIVER BIG: no, yes
    10. LIVER FIRM: no, yes
    11. SPLEEN PALPABLE: no, yes
    12. SPIDERS: no, yes
    13. ASCITES: no, yes
    14. VARICES: no, yes
    15. BILIRUBIN: 0.39, 0.80, 1.20, 2.00, 3.00, 4.00
        -- see the note below
    16. ALK PHOSPHATE: 33, 80, 120, 160, 200, 250
    17. SGOT: 13, 100, 200, 300, 400, 500, 
    18. ALBUMIN: 2.1, 3.0, 3.8, 4.5, 5.0, 6.0
    19. PROTIME: 10, 20, 30, 40, 50, 60, 70, 80, 90
    20. HISTOLOGY: no, yes

     We'll return a set of 5 data partitions stratified by class.

    :param data_file: The name of the file to load the raw machine data
    :param verbose: Used to indicate whether to log detailed information
    """

    # Drop the estimation from the data set - access it separately
    # Drop labels with many missing instances
    orig_labels = ["classification", "age", "sex", "steroid", "antivirals", "fatigue", "malaise", "anorexia", "liver big", "liver firm", "spleen palpable", "spiders", "ascites", "varices", "bilirubin", "phosphate", "sgot", "albumin", "protime", "histology"]
    rearr_labels = ["classification", "age", "sex", "steroid", "antivirals", "fatigue", "malaise", "anorexia", "spleen palpable", "spiders", "ascites", "varices", "bilirubin", "sgot", "histology"]

    # Labels for features which have real values for processing
    range_labels = ["age", "bilirubin", "sgot"]

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
    createGroups("./hepatitis.data", False)