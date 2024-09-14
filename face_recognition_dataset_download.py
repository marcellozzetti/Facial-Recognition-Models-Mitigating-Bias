import os
import io
import requests
import zipfile

zip_dataset_file = 'https://dataset-fairface.s3.us-east-1.amazonaws.com/dataset/fairface-img-margin125-trainval.zip?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjELH%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIFm6Jrxzu4yYSFSbgqFVsoYFoeRaCAh1DkWTN1e3YPNdAiBMWxCg%2Fui8iexR%2FZJRArey1tj786wkOyfP9vIU0%2BYpaSrtAgjZ%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAAaDDIwNTkzMDYyMDExMyIMRhdROpG%2Bc%2BkbYN3tKsECFsztTC2%2BQPsBbFYTTSHsUuzWCawVW7CsjcF201OZHjC6oBHAe4OuVvmdYZrMaH7CUkOzI%2Bp3E3vE4gMMJvGlvBXSDZY3mD37Elkm7bsCUXTosmtoO04gLeeg%2Bfrjmzd4Ydeel%2F1%2FI9qUINTyAJIvha84gip4xqIRfbhNknssWWqpHILLSZ4HC6oNkeCkBNJKRYaCk0Vn7A0bfcsxz%2FrxbZFCcw1Y9ZvgK4DuNuOXs5qbWx9QikxYtsN1XmHKf7xZkRzzS6vpAZFfXq612AHhtGXpY9oqh5je7uBhFxcdhbZsXZOFMnvLYDwQvp6StH2WIu4xRUQjpwGkpHmnp%2F48PbC0hhkC5lpBaEW1emgyOSSNoFUzLG5%2B1jb1PAhDvdKelLh7BM%2FO4%2FIrvB2VWK7R1n2eNvZxlyv9LNdrHkQpvGS6MJTslrcGOrQCdwREjABHClupzL6dfZOO5kY73c3rqhDt8Y9wm2pfjMhBzDNGjmJpe0H0%2BrNrOfmQ1VQRwAlk2%2BSAz55xcQU3ONYhANhKc8QDlf4VD%2BW3XU3x1cQqSNyQa28Kg6cROxpu5tZiqa9L1ZcYUEuC%2FRJaiZPb5iKvRFzab72PlvK6IH15zEXm6zba9Q2HOiKjFy%2FTNQ5EsmfuycPjbdDyCCjyrhnRN4QdwvgIdBAxzO9qy7WvrURJatMFCTSi52H8lpyumocwESM18%2BnvVQLIds4CWlAh4Yl3Pd8EnnSSctliqT8mv4HNQrJqzU%2FZgQtyQEIfOkNLztGmHETu2O9jcM1HcaLAmLkKcnpMDZywZ8U1HIdJCGafbEjTn5hpKkNGf%2FwoCllFrCEYysUjEUGAiRI%2B7aWALmw%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240914T161420Z&X-Amz-SignedHeaders=host&X-Amz-Expires=43200&X-Amz-Credential=ASIAS74TL4TI3T7XUZDK%2F20240914%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=4d26a69f184cd9546994a1d30c0328122329e8dacf388be2bf7de867fbec17a6'

# Main paths
base_dir = '/home/azureuser/cloudfiles/code/Users/marcello.ozzetti'
img_base_dir = base_dir + '/fairface/dataset/'

#7. Descompactar o dataset de imagens
print("Step 6 (Imagens Download): Start")

# Checking if exists
os.makedirs(img_base_dir, exist_ok=True)

# Download the ZIP file
response = requests.get(zip_dataset_file)

print("Step 6 (Imagens Download): Dataset Downloaded")

with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
    zip_ref.extractall(img_base_dir)

print("Step 6 (Imagens Download): End")