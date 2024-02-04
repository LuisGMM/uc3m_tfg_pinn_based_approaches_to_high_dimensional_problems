
.ONESHELL:

# https://stackoverflow.com/questions/8346118/check-if-a-makefile-exists-before-including-it

.EXPORT_ALL_VARIABLES:

.PHONY: docs
docs:
	$(MAKE) -C docs/ html

.PHONY: install
install:
	: \
    && docker build -t tfg . \
	&& :

.PHONY: run
run:
	: \
    && docker run -v ./src/tfg:/tfg  --rm -it --entrypoint bash tfg \
	&& :

.PHONY: lint
lint:
	: \
    && docker run -v ./src/tfg:/tfg --rm tfg sh -c "$(lint)" \
	&& :

.PHONY: format
format:
	: \
    docker run tfg -v ./src/tfg:/tfg --rm sh -c "$(format)" \
	&& :

.PHONY: fix
fix:
	: \
    docker run tfg -v ./src/tfg:/tfg --rm sh -c "$(fix)" \
	&& :

.PHONY: test
test:
	: \
    docker run tfg -v ./src/tfg:/tfg --rm sh -c "$(test)" \
	&& :

lint := ruff check tfg tests; mypy tfg
format := ruff format tfg tests
test := pytest --cov
fix := ruff tfg tests --fix
