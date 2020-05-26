# Grupo Neoenergia 3

vai ter algo aqui e mudar o nome GRUPO Neoenergia 3 

## 📊 Dados

Antes de mais nada, para que seja possível o treinamento e a análise dos modelos de machine learning, é necessário uma **base de dados**. Para isso, utilizamos técnicas de [web scrapping](https://en.wikipedia.org/wiki/Web_scraping) para coletar de fotos de medidores de energia elétrica. Chegamos num total de 2373 fotos. Após isso, foi necessario decidir quais imagens eram realmente de um medidor de energia elétrica e estavam com qualidade boa, e quais não estavam ou não era uma imagem de medidor.

Para resolver este problema, utilizamos a técnica de [clustering](https://en.wikipedia.org/wiki/Cluster_analysis), onde foi  encontrado um total de 8 clusters. Ao realizar uma análise mais minunciosa, foi detectado que alguns clusters podiam ser agrupados como imagens boas, imagens ruins e imagens que podiam ser descartadas. Isso resultou num total de 2191 imagens, sendo destas 617 consideradas boas e 1574 consideradas ruins.

O próximo passo foi a extração de features das imagens. Para isso utilizamos a biblioteca [Keras](https://keras.io), ao qual extraiu 32768 features de uma única imagem. Como esse número é relativamente alto e o poder computacional para processar tudo isso é precisa ser levado em consideração, buscamos formas de otimizar as features que tinha uma maior corrrelação. Esse processo gerou um total de 73 features que possuiam a maior correlação.

Como queriamos aumentar a nossa base de dados, e ao tentar aumentar a quantidade de dados coletados mostrou-se ineficaz. Buscamos alternativas e encontramos a estretégia de [data augmentation](https://towardsdatascience.com/data-augmentation-for-deep-learning-4fe21d1a4eb9). Essa abordagem nós possibilitou gerar um total de 10953 imagens sem a necessidade de coletar mais dados.

Como o foco desse repositório é a análise do melhor modelo para identificação de um medidor de energia elétrica, você pode encontrar como foi feita a extração dos dados [aqui](https://github.com/Neoenergia-3/data-mining), a utilização da técnica de clustering [aqui](https://github.com/Neoenergia-3/image-clustering), a estretégia de data augmentation [aqui](https://github.com/Neoenergia-3/image-data-augmentation) e como geramos um arquivo CSV [aqui](https://github.com/Neoenergia-3/image-data-set).

## Algoritmos Utilizados

- KNN
- Regressão Logística
- Árvore de Decisão

## Análise dos Modelos

## 👩🏿‍🍳 Mão na Massa

### Pré-requisitos

- [Docker](https://docs.docker.com/get-docker/)
- [Docker-compose](https://docs.docker.com/compose/install/)

> **nota**: não é necessário a instalação de nenhuma biblioteca adicional.

### Iniciando

Primeiro, tenha certeza que está dentro do diretório do repositório:

```bash
$ cd <path/to/image-classifier>
```

Então, inicie o container:

```bash
$ docker-compose up -d # inicia o container
```

Agora, o Jupyter Notebook vai estar rodando na seguinte URL

```bash
localhost:8888
```

Quando terminar, você pode destruir o conteiner com o seguinte comando:

```zsh
$ docker-compose down # termina o container
```

## 🤝 Contribuição

Se você está interessado em ajudar a contribuir com o projeto, por favor olhe nosso [Guia de Contribuição](https://github.com/Neoenergia-3/image-classifier/CONTRIBUTING.md).

E o nosso muito obrigado a todas as [pessoas que já contribuiram](https://github.com/Neoenergia-3/image-classifier/graphs/contributors) para o projeto!

## 📝 Licença

Copyright © 2020-present, [Contribuidores](https://github.com/Neoenergia-3/image-classifier/graphs/contributors). Esse projeto é [MIT](https://github.com/lcbm/dotfiles/blob/master/LICENSE) License.
