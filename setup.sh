
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


les/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15608.jpg'): can't open/read file: check file path/integrity
[ WARN:0@18.018] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15183.jpg'): can't open/read file: check file path/integrity
[ WARN:0@18.022] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/11375.jpg'): can't open/read file: check file path/integrity
[ WARN:0@18.026] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10365.jpg'): can't open/read file: check file path/integrity
[ WARN:0@18.030] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/11774.jpg'): can't open/read file: check file path/integrity
[ WARN:0@18.034] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/11285.jpg'): can't open/read file: check file path/integrity
[ WARN:0@18.038] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/13200.jpg'): can't open/read file: check file path/integrity
[ WARN:0@18.043] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/14262.jpg'): can't open/read file: check file path/integrity
[ WARN:0@18.047] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10997.jpg'): can't open/read file: check file path/integrity
[ WARN:0@18.051] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/13752.jpg'): can't open/read file: check file path/integrity
[ WARN:0@18.056] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15466.jpg'): can't open/read file: check file path/integrity
[ WARN:0@18.060] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/13333.jpg'): can't open/read file: check file path/integrity
[ WARN:0@18.064] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/14176.jpg'): can't open/read file: check file path/integrity
/anaconda/envs/azureml_py38/lib/python3.9/site-packages/torch/amp/autocast_mode.py:265: UserWarning:

User provided device_type of 'cuda', but CUDA is not available. Disabling

[ WARN:0@18.578] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10925.jpg'): can't open/read file: check file path/integrity
[ WARN:0@18.582] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15072.jpg'): can't open/read file: check file path/integrity
[ WARN:0@18.587] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/12825.jpg'): can't open/read file: check file path/integrity
[ WARN:0@18.591] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/11748.jpg'): can't open/read file: check file path/integrity
[ WARN:0@18.595] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10617.jpg'): can't open/read file: check file path/integrity
[ WARN:0@18.598] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/14233.jpg'): can't open/read file: check file path/integrity
[ WARN:0@18.602] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/13503.jpg'): can't open/read file: check file path/integrity
[ WARN:0@18.606] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10886.jpg'): can't open/read file: check file path/integrity
[ WARN:0@18.610] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10317.jpg'): can't open/read file: check file path/integrity
[ WARN:0@18.614] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10976.jpg'): can't open/read file: check file path/integrity
[ WARN:0@18.618] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/14476.jpg'): can't open/read file: check file path/integrity
[ WARN:0@18.622] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15134.jpg'): can't open/read file: check file path/integrity
[ WARN:0@18.627] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/12968.jpg'): can't open/read file: check file path/integrity
[ WARN:0@18.632] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/11878.jpg'): can't open/read file: check file path/integrity
[ WARN:0@18.636] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15439.jpg'): can't open/read file: check file path/integrity
[ WARN:0@18.641] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/11559.jpg'): can't open/read file: check file path/integrity
[ WARN:0@18.645] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/14786.jpg'): can't open/read file: check file path/integrity
[ WARN:0@18.648] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/11287.jpg'): can't open/read file: check file path/integrity
[ WARN:0@18.653] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/11220.jpg'): can't open/read file: check file path/integrity
[ WARN:0@18.903] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15518.jpg'): can't open/read file: check file path/integrity
[ WARN:0@18.908] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/11352.jpg'): can't open/read file: check file path/integrity
[ WARN:0@18.913] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/12452.jpg'): can't open/read file: check file path/integrity
[ WARN:0@18.917] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10823.jpg'): can't open/read file: check file path/integrity
[ WARN:0@18.921] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10363.jpg'): can't open/read file: check file path/integrity
[ WARN:0@18.926] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/14199.jpg'): can't open/read file: check file path/integrity
[ WARN:0@18.930] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/11446.jpg'): can't open/read file: check file path/integrity
[ WARN:0@18.934] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15617.jpg'): can't open/read file: check file path/integrity
[ WARN:0@18.938] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/11916.jpg'): can't open/read file: check file path/integrity
[ WARN:0@18.942] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/13746.jpg'): can't open/read file: check file path/integrity
[ WARN:0@18.946] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/13118.jpg'): can't open/read file: check file path/integrity
[ WARN:0@18.950] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10694.jpg'): can't open/read file: check file path/integrity
[ WARN:0@18.955] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10354.jpg'): can't open/read file: check file path/integrity
/anaconda/envs/azureml_py38/lib/python3.9/site-packages/torch/amp/autocast_mode.py:265: UserWarning:

User provided device_type of 'cuda', but CUDA is not available. Disabling

[ WARN:0@19.214] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/12390.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.218] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15748.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.223] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15990.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.227] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10026.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.231] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15150.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.235] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/12047.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.239] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/11016.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.243] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15260.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.247] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/13575.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.251] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/12007.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.255] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/12286.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.259] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10673.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.263] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10981.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.267] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15288.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.271] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/11097.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.276] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/11279.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.280] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/12082.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.285] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15251.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.309] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15221.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.313] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/14599.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.337] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15487.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.341] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15970.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.345] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/14338.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.350] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10409.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.354] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10709.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.358] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/11333.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.362] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/13551.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.367] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/13963.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.392] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15370.jpg'): can't open/read file: check file path/integrity
/anaconda/envs/azureml_py38/lib/python3.9/site-packages/torch/amp/autocast_mode.py:265: UserWarning:

User provided device_type of 'cuda', but CUDA is not available. Disabling

[ WARN:0@19.708] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/14525.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.712] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/11488.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.716] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15256.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.720] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/12539.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.724] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/12581.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.729] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10721.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.733] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/11967.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.752] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15298.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.757] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/13015.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.761] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/11067.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.765] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/13289.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.770] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/12100.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.774] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10128.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.779] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/12417.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.812] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/14432.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.816] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15359.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.820] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/12264.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.825] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/13621.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.830] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/12209.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.834] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/11928.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.839] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/14695.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.843] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/16008.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.847] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/11820.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.872] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/13947.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.876] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15621.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.881] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15746.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.885] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/13601.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.890] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10244.jpg'): can't open/read file: check file path/integrity
[ WARN:0@19.895] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15499.jpg'): can't open/read file: check file path/integrity
/anaconda/envs/azureml_py38/lib/python3.9/site-packages/torch/amp/autocast_mode.py:265: UserWarning:

User provided device_type of 'cuda', but CUDA is not available. Disabling

[ WARN:0@20.916] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/12768.jpg'): can't open/read file: check file path/integrity
[ WARN:0@20.943] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/12831.jpg'): can't open/read file: check file path/integrity
[ WARN:0@20.979] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15807.jpg'): can't open/read file: check file path/integrity
[ WARN:0@20.983] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10649.jpg'): can't open/read file: check file path/integrity
[ WARN:0@20.987] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10999.jpg'): can't open/read file: check file path/integrity
[ WARN:0@20.992] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/13016.jpg'): can't open/read file: check file path/integrity
[ WARN:0@20.996] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/11829.jpg'): can't open/read file: check file path/integrity
[ WARN:0@21.000] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/12045.jpg'): can't open/read file: check file path/integrity
[ WARN:0@21.004] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/11292.jpg'): can't open/read file: check file path/integrity
[ WARN:0@21.008] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10326.jpg'): can't open/read file: check file path/integrity
[ WARN:0@21.012] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/11438.jpg'): can't open/read file: check file path/integrity
[ WARN:0@21.016] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10460.jpg'): can't open/read file: check file path/integrity
[ WARN:0@21.020] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/13172.jpg'): can't open/read file: check file path/integrity
[ WARN:0@21.024] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/13641.jpg'): can't open/read file: check file path/integrity
[ WARN:0@21.029] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/14067.jpg'): can't open/read file: check file path/integrity
[ WARN:0@21.033] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/13187.jpg'): can't open/read file: check file path/integrity
[ WARN:0@21.037] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/14486.jpg'): can't open/read file: check file path/integrity
[ WARN:0@21.041] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/12807.jpg'): can't open/read file: check file path/integrity
[ WARN:0@21.045] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/11400.jpg'): can't open/read file: check file path/integrity
[ WARN:0@21.049] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/13714.jpg'): can't open/read file: check file path/integrity
[ WARN:0@21.054] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/12758.jpg'): can't open/read file: check file path/integrity
[ WARN:0@21.057] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/14453.jpg'): can't open/read file: check file path/integrity
[ WARN:0@21.085] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/13079.jpg'): can't open/read file: check file path/integrity
[ WARN:0@21.090] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/13574.jpg'): can't open/read file: check file path/integrity
[ WARN:0@21.094] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/12013.jpg'): can't open/read file: check file path/integrity
[ WARN:0@21.099] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15351.jpg'): can't open/read file: check file path/integrity
[ WARN:0@21.104] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15454.jpg'): can't open/read file: check file path/integrity
[ WARN:0@21.109] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/12721.jpg'): can't open/read file: check file path/integrity
[ WARN:0@21.113] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15548.jpg'): can't open/read file: check file path/integrity
[ WARN:0@21.909] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/11291.jpg'): can't open/read file: check file path/integrity
/anaconda/envs/azureml_py38/lib/python3.9/site-packages/torch/amp/autocast_mode.py:265: UserWarning:

User provided device_type of 'cuda', but CUDA is not available. Disabling

[ WARN:0@21.915] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/11504.jpg'): can't open/read file: check file path/integrity
[ WARN:0@21.919] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/14847.jpg'): can't open/read file: check file path/integrity
[ WARN:0@21.948] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/14570.jpg'): can't open/read file: check file path/integrity
[ WARN:0@21.952] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15770.jpg'): can't open/read file: check file path/integrity
[ WARN:0@21.957] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10395.jpg'): can't open/read file: check file path/integrity
[ WARN:0@21.982] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/11718.jpg'): can't open/read file: check file path/integrity
[ WARN:0@22.006] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/12236.jpg'): can't open/read file: check file path/integrity
[ WARN:0@22.016] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15962.jpg'): can't open/read file: check file path/integrity
[ WARN:0@22.049] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/12212.jpg'): can't open/read file: check file path/integrity
[ WARN:0@22.054] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15455.jpg'): can't open/read file: check file path/integrity
[ WARN:0@22.058] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10953.jpg'): can't open/read file: check file path/integrity
[ WARN:0@22.062] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10226.jpg'): can't open/read file: check file path/integrity
[ WARN:0@22.066] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/12513.jpg'): can't open/read file: check file path/integrity
[ WARN:0@22.070] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/14677.jpg'): can't open/read file: check file path/integrity
[ WARN:0@22.075] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/11496.jpg'): can't open/read file: check file path/integrity
[ WARN:0@22.080] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10775.jpg'): can't open/read file: check file path/integrity
[ WARN:0@22.085] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10903.jpg'): can't open/read file: check file path/integrity
[ WARN:0@22.089] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/13977.jpg'): can't open/read file: check file path/integrity
[ WARN:0@22.094] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/12606.jpg'): can't open/read file: check file path/integrity
[ WARN:0@22.099] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10437.jpg'): can't open/read file: check file path/integrity
[ WARN:0@22.108] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/12292.jpg'): can't open/read file: check file path/integrity
[ WARN:0@22.112] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15847.jpg'): can't open/read file: check file path/integrity
[ WARN:0@22.116] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10787.jpg'): can't open/read file: check file path/integrity
[ WARN:0@22.121] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/11937.jpg'): can't open/read file: check file path/integrity
[ WARN:0@22.126] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10134.jpg'): can't open/read file: check file path/integrity
[ WARN:0@22.130] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/14384.jpg'): can't open/read file: check file path/integrity
[ WARN:0@22.134] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/13278.jpg'): can't open/read file: check file path/integrity
/anaconda/envs/azureml_py38/lib/python3.9/site-packages/torch/amp/autocast_mode.py:265: UserWarning:

User provided device_type of 'cuda', but CUDA is not available. Disabling

[ WARN:0@22.388] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/14402.jpg'): can't open/read file: check file path/integrity
[ WARN:0@22.391] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15020.jpg'): can't open/read file: check file path/integrity
[ WARN:0@22.396] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15717.jpg'): can't open/read file: check file path/integrity
[ WARN:0@22.401] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/13692.jpg'): can't open/read file: check file path/integrity
[ WARN:0@22.405] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10230.jpg'): can't open/read file: check file path/integrity
[ WARN:0@22.409] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10536.jpg'): can't open/read file: check file path/integrity
[ WARN:0@22.413] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15741.jpg'): can't open/read file: check file path/integrity
[ WARN:0@22.418] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/13548.jpg'): can't open/read file: check file path/integrity
[ WARN:0@22.421] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/12437.jpg'): can't open/read file: check file path/integrity
[ WARN:0@22.427] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15568.jpg'): can't open/read file: check file path/integrity
[ WARN:0@22.431] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/14039.jpg'): can't open/read file: check file path/integrity
[ WARN:0@22.435] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10690.jpg'): can't open/read file: check file path/integrity
[ WARN:0@22.440] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15075.jpg'): can't open/read file: check file path/integrity
[ WARN:0@22.444] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15349.jpg'): can't open/read file: check file path/integrity
[ WARN:0@22.448] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10768.jpg'): can't open/read file: check file path/integrity
[ WARN:0@22.452] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/11737.jpg'): can't open/read file: check file path/integrity
[ WARN:0@22.457] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/12070.jpg'): can't open/read file: check file path/integrity
[ WARN:0@22.461] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/11872.jpg'): can't open/read file: check file path/integrity
[ WARN:0@22.465] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/13897.jpg'): can't open/read file: check file path/integrity
[ WARN:0@22.470] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10442.jpg'): can't open/read file: check file path/integrity
[ WARN:0@22.474] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15649.jpg'): can't open/read file: check file path/integrity
[ WARN:0@22.478] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10121.jpg'): can't open/read file: check file path/integrity
[ WARN:0@22.534] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10077.jpg'): can't open/read file: check file path/integrity
[ WARN:0@22.555] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10552.jpg'): can't open/read file: check file path/integrity
[ WARN:0@22.560] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10686.jpg'): can't open/read file: check file path/integrity
[ WARN:0@22.565] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/13355.jpg'): can't open/read file: check file path/integrity
[ WARN:0@22.569] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/14866.jpg'): can't open/read file: check file path/integrity
[ WARN:0@22.613] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/12928.jpg'): can't open/read file: check file path/integrity
/anaconda/envs/azureml_py38/lib/python3.9/site-packages/torch/amp/autocast_mode.py:265: UserWarning:

User provided device_type of 'cuda', but CUDA is not available. Disabling

[ WARN:0@23.183] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/12301.jpg'): can't open/read file: check file path/integrity
[ WARN:0@23.187] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/14288.jpg'): can't open/read file: check file path/integrity
[ WARN:0@23.191] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/12259.jpg'): can't open/read file: check file path/integrity
[ WARN:0@23.196] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/14935.jpg'): can't open/read file: check file path/integrity
[ WARN:0@23.199] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/12273.jpg'): can't open/read file: check file path/integrity
[ WARN:0@23.203] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10993.jpg'): can't open/read file: check file path/integrity
[ WARN:0@23.207] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/13242.jpg'): can't open/read file: check file path/integrity
[ WARN:0@23.211] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/14864.jpg'): can't open/read file: check file path/integrity
[ WARN:0@23.234] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/12482.jpg'): can't open/read file: check file path/integrity
[ WARN:0@23.239] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15641.jpg'): can't open/read file: check file path/integrity
[ WARN:0@23.243] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/13008.jpg'): can't open/read file: check file path/integrity
[ WARN:0@23.247] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/12995.jpg'): can't open/read file: check file path/integrity
[ WARN:0@23.290] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/12984.jpg'): can't open/read file: check file path/integrity
[ WARN:0@23.295] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15480.jpg'): can't open/read file: check file path/integrity
[ WARN:0@23.299] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/13677.jpg'): can't open/read file: check file path/integrity
[ WARN:0@23.303] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/14945.jpg'): can't open/read file: check file path/integrity
[ WARN:0@23.307] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/12829.jpg'): can't open/read file: check file path/integrity
[ WARN:0@23.312] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/12166.jpg'): can't open/read file: check file path/integrity
[ WARN:0@23.316] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/12405.jpg'): can't open/read file: check file path/integrity
[ WARN:0@23.321] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15069.jpg'): can't open/read file: check file path/integrity
[ WARN:0@23.326] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/11848.jpg'): can't open/read file: check file path/integrity
[ WARN:0@23.330] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/11452.jpg'): can't open/read file: check file path/integrity
[ WARN:0@23.367] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10080.jpg'): can't open/read file: check file path/integrity
[ WARN:0@23.390] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/11222.jpg'): can't open/read file: check file path/integrity
[ WARN:0@23.395] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10398.jpg'): can't open/read file: check file path/integrity
[ WARN:0@23.399] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10998.jpg'): can't open/read file: check file path/integrity
[ WARN:0@23.404] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/13035.jpg'): can't open/read file: check file path/integrity
[ WARN:0@23.408] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/11793.jpg'): can't open/read file: check file path/integrity
/anaconda/envs/azureml_py38/lib/python3.9/site-packages/torch/amp/autocast_mode.py:265: UserWarning:

User provided device_type of 'cuda', but CUDA is not available. Disabling

[ WARN:0@24.015] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/12240.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.020] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/11370.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.024] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10594.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.048] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10852.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.053] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15597.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.057] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/14182.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.060] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/14726.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.064] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15947.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.068] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15859.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.073] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15193.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.077] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/16005.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.081] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/11873.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.086] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/12243.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.090] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/13330.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.094] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10669.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.098] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/11594.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.103] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15333.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.106] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/14080.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.110] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/12050.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.115] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10684.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.119] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15149.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.123] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/12507.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.127] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/12391.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.131] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/12933.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.182] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/14688.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.186] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/12472.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.190] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/12673.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.196] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/13839.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.200] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10913.jpg'): can't open/read file: check file path/integrity
/anaconda/envs/azureml_py38/lib/python3.9/site-packages/torch/amp/autocast_mode.py:265: UserWarning:

User provided device_type of 'cuda', but CUDA is not available. Disabling

[ WARN:0@24.447] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10856.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.452] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10440.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.456] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/11880.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.460] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/12300.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.464] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10206.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.468] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/10375.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.472] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/13022.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.476] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/14380.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.530] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/13936.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.534] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15139.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.537] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15130.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.541] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/12605.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.546] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/12115.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.550] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/11418.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.555] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15671.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.559] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/11509.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.578] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15974.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.583] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/14504.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.587] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/14675.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.590] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/11990.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.594] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/11624.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.802] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/14564.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.849] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/14671.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.853] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/13090.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.858] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15541.jpg'): can't open/read file: check file path/integrity
[ WARN:0@24.884] global loadsave.cpp:241 findDecoder imread_('/home/azureuser/cloudfiles/code/Users/marcello.ozzetti/fairface/dataset/output/processed_images/train/15913.jpg'): can't open/read file: check file path/integrity
Traceback (most recent call last):
  File "/mnt/batch/tasks/shared/LS_root/mounts/clusters/fairfacetraining/code/Users/marcello.ozzetti/Facial-Recognition-Models-Mitigating-Bias/main.py", line 243, in <module>
    outputs = model(images)
  File "/anaconda/envs/azureml_py38/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/anaconda/envs/azureml_py38/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/anaconda/envs/azureml_py38/lib/python3.9/site-packages/torch/nn/parallel/data_parallel.py", line 168, in forward
    return self.module(*inputs, **kwargs)
  File "/anaconda/envs/azureml_py38/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/anaconda/envs/azureml_py38/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/mnt/batch/tasks/shared/LS_root/mounts/clusters/fairfacetraining/code/Users/marcello.ozzetti/Facial-Recognition-Models-Mitigating-Bias/main.py", line 171, in forward
    x = self.backbone(x)
  File "/anaconda/envs/azureml_py38/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/anaconda/envs/azureml_py38/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/anaconda/envs/azureml_py38/lib/python3.9/site-packages/torchvision/models/resnet.py", line 285, in forward
    return self._forward_impl(x)
  File "/anaconda/envs/azureml_py38/lib/python3.9/site-packages/torchvision/models/resnet.py", line 268, in _forward_impl
    x = self.conv1(x)
  File "/anaconda/envs/azureml_py38/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/anaconda/envs/azureml_py38/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/anaconda/envs/azureml_py38/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 458, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/anaconda/envs/azureml_py38/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 454, in _conv_forward
    return F.conv2d(input, weight, bias, self.stride,
RuntimeError: Expected 3D (unbatched) or 4D (batched) input to conv2d, but got input of size: [0]
(azureml_py38) azureuser@fairfacetraining:~/cloudfiles/code/Users/marcello.ozzetti/Facial-Recognition-Models-Mitigating-Bias$ 
(azureml_py38) azureuser@fairfacetraining:~/cloudfiles/code/Users/marcello.ozzetti/Facial-Recognition-Models-Mitigating-Bias$ python main.py 
Step 1 (Imports): Starting
/anaconda/envs/azureml_py38/lib/python3.9/site-packages/torchvision/io/image.py:13: UserWarning: Failed to load image Python extension: libtorch_cuda_cu.so: cannot open shared object file: No such file or directory
  warn(f"Failed to load image Python extension: {e}")
2024-09-14 20:37:23.349656: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /anaconda/envs/azureml_py38/lib/python3.9/site-packages/cv2/../../lib64:
2024-09-14 20:37:23.349694: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
Step 1 (Imports): Starting
Step 1 (Imports): End
Step 2 (Global Variables): Start
2024-09-14 20:37:24.513146: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcuda.so.1
2024-09-14 20:37:29.101680: E tensorflow/stream_executor/cuda/cuda_driver.cc:328] failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
2024-09-14 20:37:29.101727: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (fairfacetraining): /proc/driver/nvidia/version does not exist
2024-09-14 20:37:29.102011: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
device:  cpu
Step 2 (Global Variables): End
Step 3 (Join Dataset): Start
Step 3 (Join Dataset): End
Step 4 (Import dataSet): Start
Dataset size:
Rows: 97698
Columns: 5

Statistical Overview:
               file    age gender   race service_test
count         97698  97698  97698  97698        97698
unique        97698      9      2      7            2
top     train/1.jpg  20-29   Male  White        False
freq              1  28898  51778  18612        52284

Checking null values:
file            0
age             0
gender          0
race            0
service_test    0
dtype: int64

Column Class Distribuition 'age':
age
20-29           28898
30-39           21580
40-49           12097
3-9             11764
10-19           10284
50-59            7024
60-69            3100
0-2              1991
more than 70      960
Name: count, dtype: int64

Column Class Distribuition 'gender':
gender
Male      51778
Female    45920
Name: count, dtype: int64

Column Class Distribuition 'race':
race
White              18612
Latino_Hispanic    14990
East Asian         13837
Indian             13835
Black              13789
Southeast Asian    12210
Middle Eastern     10425
Name: count, dtype: int64
Step 4 (Import dataSet): End
Step 5 (Imagens Functions): Start
Step 5 (Imagens Functions): End
Step 7 (Images Adjustments): Start
Step 7 (Images Adjustments): End
Step 8 (Rebalance Dataset): Start
Sample count per class before balancing:
race
White              18571
Latino_Hispanic    14959
Indian             13812
East Asian         13803
Black              13761
Southeast Asian    12189
Middle Eastern     10407
Name: count, dtype: int64

Sample count per class after balancing:
race
White              10407
Latino_Hispanic    10407
Indian             10407
East Asian         10407
Black              10407
Southeast Asian    10407
Middle Eastern     10407
Name: count, dtype: int64
Step 8 (Rebalance Dataset): End
Step 1 (Imports): End
Step 9 (CNN model): Start
               file    age gender             race service_test
count          5000   5000   5000             5000         5000
unique         5000      9      2                7            2
top     train/1.jpg  20-29   Male  Latino_Hispanic        False
freq              1   1460   2652              736         2628
label_encoder  LabelEncoder()
