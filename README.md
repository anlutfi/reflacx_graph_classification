## Representando Dados do REFLACX como grafos

### Detalhamento do REFLACX

A figura abaixo ilustra a estrutura do REFLACX. Cada exame de imagem da base original (MIMIC-CXR) pode gerar várias observações, cada uma gerando um datapoint no REFLACX.
Cada ponto do REFLACX contem 5 atributos relevantes, em verde na imagem. São eles: Uma série de rótulos binários para descrever a presença de anomalias; o caminho do olhar do médico; a transcrição da voz do médico durante o processo; as coordenadas do retângulo que contem o tórax na radiografia original; e as coordenadas das elipses que contém as anomalias presentes.

O trabalho de enriquecimento de dados realizado até o momento --em azul-- foi o de dividir a transcrição em frases e, analogamente, dividir as fixações do olhar para cada uma destas frases.

![](reflacx_structure.png)

O próximo passo nesta pesquisa é tratar cada um desses grupos de fixações por frase como um grafo, para que o diagnóstico das anomalias seja aprendido por uma Graph Convolutional Network(GCN).

Para facilitar o entendimento, segue uma descrição em mais detalhes a estrutura de cada fixação do olhar na base do REFLACX:
1. timestamps de início e final da fixação
2. posição (x,y) da fixação no espaço da radiografia inteira, e não da tela
3. área medida da pupila do radiologista
4. resoluções angulares, vertical e horizontal, da radiografia para o nível de zoom expecífico da fixação. Ou seja, quantos pixels são observados por grau do cone de visão do radiologista
5. métricas da janela do software de observação
6. as coordenadas do crop do exame exibido na tela no espaço da imagem inteira
7. a posição deste crop em coordenadas da tela

Para a modelagem desta estrutura de dados, descrita mais adiante, foi necessário considerar o que seria importante para um vértice, bem como nos critérios para que dois vértices sejam considerados vizinhos, partilhando uma aresta.

### Estrutura de Grafo Proposta

#### Vértices

Para o vértice em si, 3 grandezas foram consideradas relevantes: o tempo que o radiologista repousou o olhar; a posição daquela observação em relação ao tórax; e a região da imagem que se observou naquela fixação. O tempo se obtem facilmente pelo atributo 1 descrito acima. A posição relativa ao tórax pode ser obtida transformando a posição do espaço da imagem para o espaço da bounding box do torax, normalizando-a para [0,1]. Já a região da imagem considerada observada é importante para feature extraction. Seguindo o método do próprio código do REFLACX para a geração de heatmaps, a região observada em uma determinada fixação é representada por uma distribuição normal com média igual à posição da fixação e desvio padrão igual resolução angular (item 4 acima). Sendo assim, o recorte da radiografia a ser usado para representar uma fixação pode ser definido por N = número de desvios padrões que se deseja considerar.

![](nodes.png)

### Feature Extraction

O modelo utilizado como feature extractor é o densenet121-res224-mimic_ch (https://github.com/mlmed/torchxrayvision, Cohen2022xrv, chexpert: irvin2019chexpert)


```python
import numpy as np
import matplotlib.pyplot as plot
import torch
```


```python
full_meta_path = '../reflacx_lib/full_meta.json' # if file doesn't exist, it will be created
reflacx_dir = "../data/reflacx"
mimic_dir = "../data/mimic/reflacx_imgs"

from metadata import Metadata

metadata = Metadata(reflacx_dir, mimic_dir, full_meta_path)
```

    loading metadata
    metadata loaded from file



```python
dicom_id = '0658ad3c-b4f77a56-2ed1609f-ea71a443-d847a975'
reflacx_id = 'P109R167865'
sample = metadata.get_sample(dicom_id, reflacx_id)
```


```python
from feature_extraction.dense_feature_extraction import DenseFeatureExtractor
extractor = DenseFeatureExtractor()
```


```python
features = extractor.get_reflacx_img_features(sample)
```


```python
features.shape
```




    torch.Size([1024, 7, 7])



#### Feature Extraction para cada Fixation

As features acima representam a imagem inteira. Mais especificamente, cada um dos 7 X 7 tensores divide a imagem em 49 regiões e descreve cada uma delas. Existem 1024 destes tensores 7X7 para cada imagem


```python
example = features[0].tolist()
for line in example:
    for n in line:
        print("{:.3f} ".format(n), end='')
    print('\n')
```

    -0.039 -0.039 -0.040 -0.043 -0.042 -0.041 -0.038 
    
    -0.044 -0.044 -0.044 -0.045 -0.043 -0.043 -0.045 
    
    -0.044 -0.042 -0.044 -0.045 -0.042 -0.043 -0.044 
    
    -0.042 -0.043 -0.041 -0.043 -0.043 -0.043 -0.043 
    
    -0.043 -0.043 -0.042 -0.046 -0.043 -0.043 -0.040 
    
    -0.045 -0.045 -0.048 -0.050 -0.048 -0.041 -0.040 
    
    -0.044 -0.050 -0.053 -0.053 -0.052 -0.047 -0.042 
    


![](feature_grid.png)

Se torna necessário determinar, para cada fixation e sua respectiva região de atenção, quais regiões considerar

![](feature_grid_fixation.png)

O crop da fixation está localizado em mais de uma região (quanto mais distante for o zoom, mais regiões). Para determinar um array de features para este ponto, faz-se uma média das features de cada região que o compreende, ponderada pelas áreas correspondentes. O resultado é um tensor de 1024 X 1, do mesmo formato das features extraídas da DensNet, porém só considerando as regiões relevantes para uma fixation


```python
fixations_features = extractor.get_all_fixations_features(sample)
```


```python
fixations_features[10].shape
```




    (1024,)



#### Arestas

Já para as arestas, outras 3 medidas foram consideradas até o momento:
1. Gaze Path Edges: a posição de uma fixação na sequência de fixações do caminho do olhar. Uma fixação é vizinha da anterior e da próxima
2. Euclidean Edges: a distância euclideana entre duas observações, com peso da aresta maior para vértices mais próximos. Como cada vértice possui sua posição (x, y) normalizada para [0, 1], o peso da aresta é 2^0.5 - distância.
3. IOU edges: a similaridade entre as regiões da imagem obtidas entre duas fixações. Dois vértices podem estar próximos em distância, mas, devido a possíveis diferenças do nível de zoom a cada fixação, não possuirem muita área da imagem em comum. O peso da aresta é dado pela IOU das imagens de ambos os vértices

![](edges.png)

O grafo terá arestas de todos os três tipos, com 3 matrizes de adjacência. Talvez seja interessante considerar uma forma de consolidá-las em uma só para a implementação.


### Preocupações Antecipadas

A estrutura proposta sugere alguns pontos de preocupação que devem ser estudados durante sua implementação

#### Orientação das arestas de Gaze Path
Embora a abordagem mais simples para este tipo de aresta seja considerá-la não direcionada, existe um ponto a favor de um direcionamento. Uma fixação no instante t esclarece uma questão levantada em instantes ateriores, bem como levanta questões que serão esclarecidas em instantes futuros. Portanto, pode ser que seja importante diferenciar a "ida" da "volta", ou seja: talvez o peso da aresta que levanta questões seja diferente do peso da aresta que às esclarece.

#### Redundância entre as relações euclideanas e as de IOU
Como o intervalo entre as fixações tende a ser pequeno, é possível que a distância euclideana entre dois vértices esteja fortemente correlacionada com a área de IOU entre eles. Se este for o caso, seria como se o modelo considerasse os mesmos parâmetros duas vezes. Faz-se necessário um teste para medir esta correlação e, então, decidir se vale a pena manter as duas métricas.
