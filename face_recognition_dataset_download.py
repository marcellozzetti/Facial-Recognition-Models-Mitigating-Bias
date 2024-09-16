import os
import io
import requests
import zipfile

zip_dataset_file = 'https://dataset-fairface.s3.us-east-1.amazonaws.com/dataset/processed_images.zip?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjENH%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJIMEYCIQCNjfEf1da7tjRsCgDkUiw2T%2Fh4u4HL2xSbdV6ox%2FyHUQIhAKh%2Frse2b%2F3IY6OZm4JHeVb0UVBBYISzO2cr32JwgUnPKu0CCPr%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQABoMMjA1OTMwNjIwMTEzIgyt3T%2FP7ZpDlGQ2JYEqwQLj9JZvy4rbnFnO6lWDF6PgjQ775GQjhsor%2BYuIEBXlEXvF40DKZvvO8kEXBL7d93OexFn%2Fv4%2Fz7Ku6tOybNGwmWsglBCl449d82xbVZXG8x6cPUOFVHek%2FOosYZvElX%2BA9aCHb0kobLPdOGRpMsJvl%2FlKAj4HDJt%2FVU0%2Bi%2FZpYNR%2BGecBdpO2GL%2BsCBiqJGBYLibH7UgQeXj0HLt15jJS13I1hlZBSpX%2Fg7ULtPgBTtvcrQQBlJlEQqIhg7aT0dSnFict5JVhNfOd6XjJ0O65d7UWUU80HQ0HMVcmI389O3RkfIAr4i111B32FjPg2DfU%2F0FuHmV9jgvltlWt%2BF%2B4HYArSKVSQ9nGAXmCFp29DZLOx7Sp%2B9UMuK50PnK8dvGDvL%2FNCid7611kq%2B3IEPZNbOjIpaXXSNyUEUqFASxge7Gsw34KetwY6sgI91KAA%2B4qh%2FnEbM2pM%2FVNPhhjLKsIFlgVnKs8NXr9u0QM4L8gHxjyginSccdVP%2FNl0M1D%2FZPnb71AN%2BSvQa9DxaZpcz81z4Zokxmw0ZIJzDllVl0qebx2jfJ8UNuJsXh43Y5nGQrr4yN2z1GSBUi2hwPxUoDNuTogWUat7huqqOFEA1i3mCJwevSuJp5ZqYng0VScfSmFtPzXtbg51aNKzPX%2BoXkBNA7M3FrgMm6yQ6Cjzn9K2t1OefePPRklTQhwEJLj7HN78UCwpSDkuG5k9PpXRVRsKw2nlRPX9wOXbapxRY%2FL6cYw0fvCMX2%2BdIEp9MtQ3h0e5lLCD5shsGIK5huTOY%2FYeq21OHhAVX0qubzK%2F78shL2VmEJCorncY5czwaCKYLl2mg3l9tqgE8uy5Edg%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240916T010843Z&X-Amz-SignedHeaders=host&X-Amz-Expires=14400&X-Amz-Credential=ASIAS74TL4TIWZ2WMBXV%2F20240916%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=3ded1951af193cc6fec2daf66b2fb5242b76682d4e943411bef5f949a9b56f8f'

# Main paths
base_dir = '/home/ubuntu/projects/'
img_base_dir = base_dir + 'fairface/dataset/output/processed_images/'

print("Step 6 (Images Download): Start")

# Checking if exists
os.makedirs(img_base_dir, exist_ok=True)

# Download the ZIP file
response = requests.get(zip_dataset_file)

print("Step 6 (Images Download): Dataset Downloaded")

with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
    zip_ref.extractall(img_base_dir)

print("Step 6 (Images Download): End")