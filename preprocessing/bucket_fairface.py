import boto3
from botocore.exceptions import NoCredentialsError

aws_access_key_id = 'None' 
aws_secret_access_key = 'None' 

def upload_to_s3(file_name, bucket, object_name=None):
    if object_name is None:
        object_name = file_name

    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
        )
    
    try:
        s3_client.upload_file(file_name, bucket, object_name)
        print(f"Arquivo {file_name} carregado com sucesso para {bucket}/{object_name}")
    except FileNotFoundError:
        print(f"O arquivo {file_name} não foi encontrado.")
    except NoCredentialsError:
        print("Credenciais não disponíveis.")
    except Exception as e:
        print(f"Ocorreu um erro: {str(e)}")

if __name__ == "__main__":
    file_path = "/home/ubuntu/projects/fairface/dataset/output/processed_images/"
    file_name = "fairface-processed_images.zip"
    bucket_name = "dataset-fairface"
    object_name = "dataset/fairface-processed_images.zip"

    upload_to_s3(file_path + file_name, bucket_name, object_name)

