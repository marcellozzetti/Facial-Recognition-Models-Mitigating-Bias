
# Update SO
sudo apt update -y

ssh -i fairface-training-gpu_key.pem azureuser@172.212.201.132

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

#Install GitHub Client
sudo apt install gh

gh auth login

#Token: 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

#Add main directory
mkdir projects

cd projects/

# Downloading the main repository
gh repo clone marcellozzetti/Facial-Recognition-Models-Mitigating-Bias

cd Facial-Recognition-Models-Mitigating-Bias/

# Install used libs Python

pip3 install pandas

pip3 install seaborn

pip3 install plotly

pip3 install torchvision

#pip install facenet-pytorch

pip3 install scikit-learn

pip3 install MTCNN

pip3 install tensorflow

pip3 install tensorflow-gpu


https://ml.azure.com/fileexplorerAzNB?wsid=/subscriptions/c0522196-e734-4fc8-8ccb-d1c6a73ff363/resourceGroups/marcello.ozzetti-rg/providers/Microsoft.MachineLearningServices/workspaces/FairFace-Training&tid=ac78bd89-c96e-4aa5-9c11-111c7f465624&activeFilePath=Terminals/FairfaceTraining/2