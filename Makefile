
.ONESHELL:

# https://stackoverflow.com/questions/8346118/check-if-a-makefile-exists-before-including-it

.EXPORT_ALL_VARIABLES:

.PHONY: docs
docs:
	$(MAKE) -C docs/ html

.PHONY: docker-install
docker-install:
	: \
    && docker build -t tfg . \
	&& :


.PHONY: docker-run
docker-run:
	: \
    && docker run -v ./src/tfg:/tfg  --rm -it --entrypoint bash tfg \
	&& :

.PHONY: docker-lint
docker-lint:
	: \
    && docker run -v ./src/tfg:/tfg --rm tfg sh -c "$(lint)" \
	&& :

.PHONY: docker-format
docker-format:
	: \
    && docker run tfg -v ./src/tfg:/tfg --rm sh -c "$(format)" \
	&& :

.PHONY: docker-fix
docker-fix:
	: \
    && docker run tfg -v ./src/tfg:/tfg --rm sh -c "$(fix)" \
	&& :

.PHONY: docker-test
docker-test:
	: \
    && docker run tfg -v ./src/tfg:/tfg --rm sh -c "$(test)" \
	&& :


.PHONY: install
install:
	: \
	&& virtualenv -p python3.11 venv \
	&& echo "../../../../src/" > venv/lib/python3.11/site-packages/tfg.pth \
	&& . venv/bin/activate \
	&& pip install -r ./requirements/dev.txt \
	&& :

.PHONY: lint
lint:
	: \
	&& . venv/bin/activate \
        && ruff check src/tfg tests; mypy src/tfg \
	&& :

.PHONY: format
format:
	: \
	&& . venv/bin/activate \
        && ruff format src/tfg tests \
	&& :

.PHONY: fix
fix:
	: \
	&& . venv/bin/activate \
        && ruff src/tfg tests --fix \
	&& :

.PHONY: test
test:
	: \
	&& . venv/bin/activate \
        && pytest --cov \
	&& :

lint := ruff check tfg tests; mypy tfg
format := ruff format tfg tests
test := pytest --cov
fix := ruff tfg tests --fix
