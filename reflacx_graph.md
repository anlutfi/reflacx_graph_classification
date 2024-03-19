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

#### Vértices

Para o vértice em si, 3 grandezas foram consideradas relevantes: o tempo que o radiologista repousou o olhar; a posição daquela observação em relação ao tórax; e a região da imagem que se observou naquela fixação. O tempo se obtem facilmente pelo atributo 1 descrito acima. A posição relativa ao tórax pode ser obtida transformando a posição do espaço da imagem para o espaço da bounding box do torax, normalizando-a para [0,1]. Já a região da imagem considerada observada é importante para feature extraction. Seguindo o método do próprio código do REFLACX para a geração de heatmaps, a região observada em uma determinada fixação é representada por uma distribuição normal com média igual à posição da fixação e desvio padrão igual resolução angular (item 4 acima). Sendo assim, o recorte da radiografia a ser usado para representar uma fixação pode ser definido por N = número de desvios padrões que se deseja considerar.

![](nodes.png)

#### Arestas

Já para as arestas, outras 3 medidas foram consideradas até o momento:
1. a posição de uma fixação na sequência de fixações do caminho do olhar. Uma fixação é vizinha da anterior e da próxima
2. a distância euclideana entre duas observações, com peso da aresta maior para vértices mais próximos. Como cada vértice possui sua posição (x, y) normalizada para [0, 1], o peso da aresta é 2^0.5 - distância.
3. a similaridade entre as regiões da imagem obtidas entre duas fixações. Dois vértices podem estar próximos em distância, mas, devido a possíveis diferenças do nível de zoom a cada fixação, não possuirem muita área da imagem em comum. O peso da aresta é dado pela IOU das imagens de ambos os vértices

![](edges.png)

O grafo terá arestas de todos os três tipos, com 3 matrizes de adjacência. Talvez seja interessante considerar uma forma de consolidá-las em uma só para a implementação.

