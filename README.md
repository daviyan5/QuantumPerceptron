# QuantumPerceptron
*Projeto da disciplina de Algoritmos de Avançados (IF775)*
Uma implementação do Perceptron Linear utilizando circuitos quânticos.

----

#### Dependências
_Python 3_ é requisitado para a execução.
As bibliotecas _Qiskit_ e _Numpy_ são necessárias.
Uma GPU com _CUDA_ é altamente recomendada.

#### Execução
O treinamento do modelo pode ser efetuado por:

```bash
make train
```
ou

```bash
make train_reload
```
Para recarregar o database.

#### Testes
Imagens para classificação devem ser colocadas na pasta "./src/data/other".
Para executar com os testes:

```bash
make test
```

Mais detalhes sobre as opções podem ser adquiridos com 


```bash
make help
```

