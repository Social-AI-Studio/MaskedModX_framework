### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Import libraries-------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
import tarfile
import urllib.request

import pandas as pd

### --------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------Download SBIC dataset----------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
urllib.request.urlretrieve(
    "https://maartensap.com/social-bias-frames/SBIC.v2.tgz",
    "SBIC.v2.tgz"
)

### --------------------------------------------------------------------------------------------------------- ###
### ----------------------------------------Uncompress SBIC .tgz file---------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
# Uncompress a .tgz file. via
# https://www.geeksforgeeks.org/how-to-uncompress-a-tar-gz-file-using-python/
file = tarfile.open("SBIC.v2.tgz")
file.extractall("SBIC.v2")
file.close()

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------Aggregate offensiveYN per post for test dataset------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
df = pd.read_csv(
    filepath_or_buffer="SBIC.v2/SBIC.v2.tst.csv",
    encoding="utf-8"
)
# Uncomment this line for accurate computation of dataset statistics (modal count of Yes, Maybe, No, Incomprehensible).
# df = df.drop_duplicates(subset=["WorkerId", "HITId", "annotatorGender", "annotatorRace", "annotatorPolitics", "annotatorAge"], keep="first").reset_index(drop=True)
# Reverse the list (replace with ["Incomprehensible", "No", "Maybe", "Yes"]) to reverse the priority of tie-breaking.
offensiveYN_aggregated_test = (
    df.groupby("post")["offensiveYN"]
      .apply(lambda grp: grp.map({1.0: "Yes", 0.5: "Maybe", 0.0: "No"}).fillna("Incomprehensible")
             .value_counts()
             .reindex(["Yes", "Maybe", "No", "Incomprehensible"], fill_value=0)
             .idxmax())
      .reset_index(name="offensiveYN_mode")
)

### --------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------Save aggregated file------------------------------------------ ###
### --------------------------------------------------------------------------------------------------------- ###
offensiveYN_aggregated_test.to_csv(
    "offensiveYN_aggregated_test.csv",
    index=False,
    encoding="utf-8"
)
