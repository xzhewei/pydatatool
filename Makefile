DEV_USER_FLAG=$(shell python -c "import sys; print('' if hasattr(sys, 'real_prefix') else '--user')")

.PHONY: default
default: dev

install:
	python setup.py install

.PHONY: dev
dev:
	python setup.py develop $(DEV_USER_FLAG)

.PHONY: clean
clean:
	python setup.py develop --uninstall $(DEV_USER_FLAG)
	rm -rf build