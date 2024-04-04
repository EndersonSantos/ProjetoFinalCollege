build:
	docker build --platform linux/amd64 --progress plain . -t projetofinal

lint:
	docker run projetofinal /bin/bash -c "pip install -r requirements.txt && black --check projetofinal/"

test:
	docker run projetofinal /bin/bash -c "pip install -r requirements.txt && pytest"

debug:
	@docker run -it -v $(shell pwd)/projetofinal:/projetofinal/projetofinal projetofinal /bin/bash



.PHONY: build lint test