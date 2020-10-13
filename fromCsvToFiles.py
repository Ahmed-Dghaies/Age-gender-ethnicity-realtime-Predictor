import os
import os.path
import csv
from PIL import Image


# Ethnicity labels
ETHNICITIES = { 
    0: "White", 
    1: "Black",
    2: "Asian",
    3: "Indian",
    4: "Hispanic"
}

# Gender labels
GENDERS = { 
    0: "Male", 
    1: "Female"
}

AGES = {
    0: "Child",
    1: "Teen",
    2: "Young_Adult",
    3: "Adult",
    4: "Elderly"
}


# Directory to store the images
base_path = os.path.join(os.path.curdir, 'images')
labeled_paths = {}
if not os.path.isdir(base_path):
    os.makedirs(base_path)

# Sort images in directories by ethnicity then gender
# ToDo: Add ages labes and directories 
for ek, ev in ETHNICITIES.items():
    for ak, av in AGES.items():
        for gk, gv in GENDERS.items():     
            labeled_path = os.path.join(base_path, ev, av, gv)
            if (gk == 0):
                labeled_paths.setdefault(ek,{})[ak] = {0: "", 1: ""}
            labeled_paths.setdefault(ek,{})[ak][gk] = labeled_path
            if not os.path.isdir(labeled_path):
                os.makedirs(labeled_path)


with open("age_gender.csv","r") as fp:
    # Read dataset as dict entries
    csv_reader = csv.DictReader(fp)

    for row in csv_reader:
        age = int(row['age'])
        ethnicity = int(row['ethnicity'])
        gender = int(row['gender'])

        ageLabel = ""

        if age > 61:
            ageLabel = 4
        elif age > 21:
            ageLabel = 3
        elif age > 18:
            ageLabel = 2
        elif age > 12:
            ageLabel = 1
        else: ageLabel = 0


        # Extract pixel string (string list of grayscale integers sep by space)
        pixels = bytearray([int(px) for px in row['pixels'].split(' ')])

        # Create new Image of resolution 48*48 from pixels
        #   I assumed a square image, so I computed
        #     len(pixels)**.5  # = 48.0
        #   to get the resolution.
        img = Image.frombytes('L', (48,48), bytes(pixels))

        # Name of file to write to
        #file_name = row['img_name']
        # or name file with labels: age_ethnicity_gender-original.jpg
        file_name = f"{ageLabel}_{ethnicity}_{gender}-{row['img_name'].split('.')[0]}.jpg"

        # The path to save the image to
        file_dir = labeled_paths.get(ethnicity, {}).get(ageLabel, {}).get( gender, base_path)
        file_path = os.path.join(file_dir, file_name)

        # Write out the Image file
        img.save(file_path)
        # or save without JPEG compression
        #img.save(file_path + ".png", 'png', compress_level=0)