build:
	docker build --platform linux/amd64 --progress plain -t final-project-college .

lint:
	docker run final-project-college /bin/bash -c "pip install -r requirements.txt && black --check projetofinal/"

test:
	docker run final-project-college /bin/bash -c "pip install -r requirements.txt && pytest"

debug:
	@docker run -it -v $(shell pwd)/projetofinal:/projetofinal/projetofinal final-project-college /bin/bash



.PHONY: build lint test