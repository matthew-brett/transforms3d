TRANSFORMS_URL=http://www.lfd.uci.edu/~gohlke/code/transformations.py

update:
	curl -O $(TRANSFORMS_URL)

make-patch:
	diff transformations.py ../transforms3d/_gohlketransforms.py \
		> transformations.py.patch; \
		if [ $$? -eq 0 ] ; then echo "no differences" ; fi

apply-patch:
	cp transformations.py ../transforms3d/_gohlketransforms.py
	cd .. && patch transforms3d/_gohlketransforms.py original/transformations.py.patch
