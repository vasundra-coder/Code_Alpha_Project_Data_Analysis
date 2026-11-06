import zipfile
import os
import glob

os.makedirs("dataset", exist_ok=True)

zip_files = glob.glob("*.zip")

if not zip_files:
    print("No ZIP file found in project folder.")

else:
    zip_path = zip_files[0]

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall("dataset")

    print("Extraction completed!")

    for root, dirs, files in os.walk("dataset"):
        for file in files:
            print(os.path.join(root, file))