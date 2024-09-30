# Definindo a arquitetura LResNet100E-IR (ResNet100 aprimorada)
class LResNet100E_IRArch(nn.Module):
    def __init__(self, num_classes):
        super(LResNet100E_IR, self).__init__()
        self.resnet = models.resnet101(weights=ResNet101_Weights.DEFAULT)
        #self.resnet = torch.hub.load('zhanghang1989/ResNeSt', 'resnest100', pretrained=True)  # Exemplo para carregar a ResNet100
        self.resnet.fc = nn.Identity()  # Mantém a saída de 2048
        
        # Definindo a camada ArcFace
        self.arcface = ArcMarginProduct(in_features=2048, out_features=num_classes)

    def forward(self, x, labels):
        features = self.resnet(x)  # Extrair características
        logits = self.arcface(features, labels)  # Calcular os logits com ArcFace
        return logits

# Definindo a camada ArcFace
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcMarginProduct, self).__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # Normalizando entrada e pesos
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        theta = torch.acos(cosine.clamp(-1.0, 1.0))
        target_logit = torch.cos(theta + self.m)

        # One-hot encoding dos rótulos
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # Combina logits com margem e ajusta escala
        output = (one_hot * target_logit) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output

# Define the model (LResNet50E-IR, a modified ResNet50 for ArcFace)
class LResNet50E_IR(nn.Module):
    def __init__(self, num_classes=len(label_encoder.classes_)):
        super(LResNet50E_IR, self).__init__()
        self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        self.backbone.fc = self.fc

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        return x

class LResNet50E_IRArc(nn.Module):
    def __init__(self, num_classes=len(label_encoder.classes_)):
        super(LResNet50E_IRArc, self).__init__()
        self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.dropout = nn.Dropout(p=0.2)
        
        # Retirar a última camada fully connected do ResNet50
        self.in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Manter as features sem aplicar FC
        
        # Camada ArcFace
        self.arc_margin = ArcMarginProduct(self.in_features, num_classes)

    def forward(self, x, labels=None):
        # Extração das features (embeddings) do backbone
        features = self.backbone(x)
        features = self.dropout(features)
        
        # Aplicar ArcFace se os rótulos estiverem disponíveis (durante o treino)
        if labels is not None:
            output = self.arc_margin(features, labels)
        else:
            output = features  # Para inferência/testes
        return output
