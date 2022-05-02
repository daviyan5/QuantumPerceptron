help:
	@echo "Para executar o treinamento do modelo, execute o comando: make train_reload, se os data sets já estiverem carregados em ./src/params, execute make train para salvar tempo";
	@echo "Para executar testes no modelo, salve as fotos desejadas em ./src/data/other, e execute make test.";
	@echo "Não execute make test sem antes garantir que existem parâmetros calculados para o modelo.";
	@echo "As dependências de execução são explicitadas no arquivo README do projeto.";
	@echo "Os parâmetros calculados são salvos em ./src/params.";
	
train:
	cd src;\
	python3 train.py;

train_reload:
	cd src;\
	python3 train.py reload;

test:
	cd src;\
	python3 test.py;
