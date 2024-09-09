import os
import requests
import zipfile

zip_dataset_file = 'https://dataset-fairface.s3.us-east-1.amazonaws.com/dataset/fairface-img-margin125-trainval.zip?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEOT%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIDFgdw8tHZTua3qFGVv8JTTfKYalK0bZnI%2BsvTnb6yJFAiAK7ZHVBULKsASufMewFxcUhuZomMuNAIuzcGUQ1csdUCrtAgj8%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAAaDDIwNTkzMDYyMDExMyIMLLt%2BKxD3qeC1ojimKsECjCttAYHj%2B%2FvuAwYMF%2BpsviCREr%2FLkwOtHce79dsOwSvmfAQCTexen45eh4R%2BJzk0%2FJtifcCOjf4rst%2Bf8WXRrMeyx8H%2FP4eSQVZJHs0sL0Cx1jvecrLEkTuE5bUZiEeje68qKG8DFAFm2G3vocVBL%2BZ0eaBqFAZZLMVG8qGkLsuleXd0wFnMpnJg8Ij8BdvwYd68lX3uYQ3e2jQzIFNuM63ije6MksZ9jHX%2BFdywtLECkSyKKniZiuSIRJ8b8ZXxK6P6X04kpqMLfDlmcW%2BHIhoato8CxuqGeJ2VaNMyGAc7rx99psBHneLtzQJh8%2FcZlnbZIGekpPm9WN5H6ucNmc8rEaGBYf9udWDsAB2TE7XkFA16x1I4GuFVW9pNErtRW%2F4S5BB8VHiO3uGsEehTvbAWJI7pVRjV0l%2B4gE04vfAAMOib57YGOrQCA8UKN6fxDia6J3h%2FySRSHNn6yHrATB2OtAMFllRwTRAdXylnVHZ4MOPwXwp4v3wCDKFylsuObIvXj8Nl2TERAFOVpeMEEmA3r%2B3ZRaJ1x%2BqMicolV5myy6uewVVeyicpLkLWSZrSYqt4QtniRVx%2F9eujH8I3JoD7hV1m%2FiGvJY6nr6hMnueVZY5A1xsHSP7XlXfpAYoGFgi0H26WWAYGKsnA%2Bq5MEudIYEKUHLUafPO3mZWd55ACtJVIzLHseVX9jq6jKLxL%2Bcp2icLdZNzAztoZCjNRsP8iOJ4ss2R1VLPhKxuwiSelZolAGkoJ2Yino%2Fasg8cVtT0Dy%2BmuFfzBbWfhvqc0o%2FYIB9nehAr1oo66I%2BfZwi0ZV65ipccsqwhetpQwleLF2QXPyW7SL0YBWMUQsNs%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240906T031808Z&X-Amz-SignedHeaders=host&X-Amz-Expires=43200&X-Amz-Credential=ASIAS74TL4TIQTEXWCAZ%2F20240906%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=b3f58c463c45dad9758ccb84538088443431917890f0e795c8afabe797204270'

# Main paths
base_dir = '/home/ubuntu/projects'
img_base_dir = base_dir + '/fairface/dataset/'

#7. Descompactar o dataset de imagens
print("Step 7 (Imagens Download): Start")

# Checking if exists
os.makedirs(img_base_dir, exist_ok=True)

# Download the ZIP file
response = requests.get(zip_dataset_file)

with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
    zip_ref.extractall(img_base_dir)

print("Step 7 (Imagens Download): End")