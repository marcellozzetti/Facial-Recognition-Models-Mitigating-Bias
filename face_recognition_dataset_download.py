import os
import io
import requests
import zipfile

def download_and_extract_dataset(url, base_dir, target_folder):
    """
    Download a ZIP file from the given URL and extract it to the specified directory.

    Args:
        url (str): URL of the ZIP file.
        base_dir (str): Base directory for extraction.
        target_folder (str): Target folder for extracted content.
    """
    # Define the full path for extraction
    img_base_dir = os.path.join(base_dir, target_folder)

    print("Step 6 (Images Download): Start")

    # Ensure the target directory exists
    os.makedirs(img_base_dir, exist_ok=True)

    # Download the ZIP file
    print("Downloading dataset...")
    response = requests.get(url)

    if response.status_code == 200:
        print("Step 6 (Images Download): Dataset downloaded successfully")
        # Extract the ZIP content
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(img_base_dir)
        print(f"Step 6 (Images Download): Files extracted to {img_base_dir}")
    else:
        print(f"Error {response.status_code}: Failed to download the dataset.")

    print("Step 6 (Images Download): End")

if __name__ == "__main__":
    # Constants for download and extraction
    ZIP_DATASET_URL = (
        'https://dataset-fairface.s3.us-east-1.amazonaws.com/dataset/processed_images.zip?response-content-disposition=inline'
        '&X-Amz-Security-Token=IQoJb3JpZ2luX2VjENH%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJIMEYCIQCNjfEf1da7tjRsCgDkUiw2T%2Fh4u4HL2xSbdV6ox%2FyHUQIhAKh%2Frse2b%2F3IY6OZm4JHeVb0UVBBYISzO2cr32JwgUnPKu0CCPr%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQABoMMjA1OTMwNjIwMTEzIgyt3T%2FP7ZpDlGQ2JYEqwQLj9JZvy4rbnFnO6lWDF6PgjQ775GQjhsor%2BYuIEBXlEXvF40DKZvvO8kEXBL7d93OexFn%2Fv4%2Fz7Ku6tOybNGwmWsglBCl449d82xbVZXG8x6cPUOFVHek%2FOosYZvElX%2BA9aCHb0kobLPdOGRpMsJvl%2FlKAj4HDJt%2FVU0%2Bi%2FZpYNR%2BGecBdpO2GL%2BsCBiqJGBYLibH7UgQeXj0HLt15jJS13I1hlZBSpX%2Fg7ULtPgBTtvcrQQBlJlEQqIhg7aT0dSnFict5JVhNfOd6XjJ0O65d7UWUU80HQ0HMVcmI389O3RkfIAr4i111B32FjPg2DfU%2F0FuHmV9jgvltlWt%2BF%2B4HYArSKVSQ9nGAXmCFp29DZLOx7Sp%2B9UMuK50PnK8dvGDvL%2FNCid7611kq%2B3IEPZNbOjIpaXXSNyUEUqFASxge7Gsw34KetwY6sgI91KAA%2B4qh%2FnEbM2pM%2FVNPhhjLKsIFlgVnKs8NXr9u0QM4L8gHxjyginSccdVP%2FNl0M1D%2FZPnb71AN%2BSvQa9DxaZpcz81z4Zokxmw0ZIJzDllVl0qebx2jfJ8UNuJsXh43Y5nGQrr4yN2z1GSBUi2hwPxUoDNuTogWUat7huqqOFEA1i3mCJwevSuJp5ZqYng0VScfSmFtPzXtbg51aNKzPX%2BoXkBNA7M3FrgMm6yQ6Cjzn9K2t1OefePPRklTQhwEJLj7HN78UCwpSDkuG5k9PpXRVRsKw2nlRPX9wOXbapxRY%2FL6cYw0fvCMX2%2BdIEp9MtQ3h0e5lLCD5shsGIK5huTOY%2FYeq21OHhAVX0qubzK%2F78shL2VmEJCorncY5czwaCKYLl2mg3l9tqgE8uy5Edg%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256'
    )
    BASE_DIR = '/home/ubuntu/projects/'
    TARGET_FOLDER = 'fairface/dataset/output/processed_images/'

    # Call the function to download and extract
    download_and_extract_dataset(ZIP_DATASET_URL, BASE_DIR, TARGET_FOLDER)
