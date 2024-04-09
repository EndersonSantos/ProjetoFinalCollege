CI_COMMIT_SHORT_SHA ?= latest

build:
	docker build --platform linux/amd64 --progress plain -t final-project-college:${CI_COMMIT_SHORT_SHA} .

lint:
	docker run final-project-college:${CI_COMMIT_SHORT_SHA} /bin/bash -c "pip install -r requirements.txt && black --check projetofinal/"

test:
	docker run final-project-college:${CI_COMMIT_SHORT_SHA} /bin/bash -c "pip install -r requirements.txt && pytest"

debug:
	@docker run -it -v $(shell pwd)/projetofinal:/projetofinal/projetofinal final-project-college:${CI_COMMIT_SHORT_SHA} /bin/bash



.PHONY: build lint test