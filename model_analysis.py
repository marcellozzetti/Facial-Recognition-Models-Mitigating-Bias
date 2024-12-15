import torch
import torch.nn as nn
from your_model_module import LResNet50E_IR  # Substitua pelo caminho correto do seu modelo

# Inicializar o modelo customizado
num_classes = 10  # Ajuste conforme o número de classes do seu problema
model = LResNet50E_IR(num_classes=num_classes)
model.eval()  # Coloca o modelo em modo de avaliação

# Função para listar todas as camadas do modelo
def listar_camadas(model):
    print("Listando as camadas do modelo:")
    for name, module in model.named_modules():
        print(name)

# Listar camadas para inspecionar
listar_camadas(model)

# Defina o nome do target_layer manualmente com base na inspeção
# Exemplo: Para LResNet50E_IR, você pode querer inspecionar uma camada do backbone, como 'backbone.layer4.2.conv3'
target_layer_name = 'backbone.layer4.2.conv3'

# Capturar a referência ao módulo da camada alvo
def obter_target_layer(model, layer_name):
    target_layer = dict(model.named_modules()).get(layer_name, None)
    if target_layer is None:
        raise ValueError(f"Camada alvo '{layer_name}' não encontrada no modelo. Verifique os nomes listados acima.")
    return target_layer

target_layer = obter_target_layer(model, target_layer_name)
print(f"\nCamada alvo encontrada: {target_layer}")

# Testar a saída da camada alvo com um tensor de entrada
# Exemplo de entrada (1 imagem, 3 canais, tamanho 224x224)
input_tensor = torch.randn(1, 3, 224, 224)

# Usar hook para capturar ativações da camada alvo
activations = None
def forward_hook(module, input, output):
    global activations
    activations = output

# Registrar o hook
hook = target_layer.register_forward_hook(forward_hook)

# Realizar a inferência
with torch.no_grad():
    _ = model(input_tensor)

# Verificar as ativações
if activations is not None:
    print("\nAtivações capturadas com sucesso:")
    print(f"Shape das ativações: {activations.shape}")
else:
    print("\nNenhuma ativação capturada. Certifique-se de que o target_layer está correto.")

# Remover o hook
hook.remove()
