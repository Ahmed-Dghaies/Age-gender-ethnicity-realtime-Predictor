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

AGES = {}

#AGE labels
for i in range(101):
    AGES[i] = str(i)


# Directory to store the images
base_path = os.path.join(os.path.curdir, 'images')
labeled_paths = {}
if not os.path.isdir(base_path):
    os.makedirs(base_path)

# Sort images in directories by ethnicity then gender
# ToDo: Add ages labes and directories 
for ek, ev in ETHNICITIES.items():
    for gk, gv in GENDERS.items():
        labeled_path = os.path.join(base_path, ev, gv)
        labeled_paths.setdefault(ek,{})[gk] = labeled_path
        if not os.path.isdir(labeled_path):
            os.makedirs(labeled_path)
print(labeled_paths)

with open("age_gender.csv","r") as fp:
    # Read dataset as dict entries
    csv_reader = csv.DictReader(fp)

    for row in csv_reader:
        age = int(row['age'])
        ethnicity = int(row['ethnicity'])
        gender = int(row['gender'])

        # Extract pixel string (string list of grayscale integers sep by space)
        pixels = bytearray([int(px) for px in row['pixels'].split(' ')])

        # Create new Image of resolution 48*48 from pixels
        #   I assumed a square image, so I computed
        #     len(pixels)**.5  # = 48.0
        #   to get the resolution.
        img = Image.frombytes('L', (48,48), bytes(pixels))

        # or name file with labels: age_ethnicity_gender-original.jpg
        file_name = f"{age:03}_{ethnicity}_{gender}-{row['img_name'].split('.')[0]}.jpg"

        # The path to save the image to
        file_dir = labeled_paths.get(ethnicity, {}).get(gender, base_path)
        file_path = os.path.join(file_dir, file_name)

        # Write out the Image file
        img.save(file_path)
        # or save without JPEG compression
        #img.save(file_path + ".png", 'png', compress_level=0)