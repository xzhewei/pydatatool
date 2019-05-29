DEV_USER_FLAG=$(shell python -c "import sys; print('' if hasattr(sys, 'real_prefix') else '--user')")

.PHONY: default
default: dev

install:
	python setup.py install

clean:
	rm -rf build

.PHONY: dev
dev:
	python setup.py develop $(DEV_USER_FLAG)

.PHONY: clean
clean:
	python setup.py develop --uninstall $(DEV_USER_FLAG)
	rm -rf build


# all:
#     # install pycocotools locally
# 	python setup.py build_ext --inplace
# 	rm -rf build
# install:
# 	# install pycocotools to the Python site-packages
# 	python setup.py build_ext install
# 	rm -rf build