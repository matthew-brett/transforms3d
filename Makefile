git-clean:
	git clean -fxd

html:
	cd doc && make html

gh-pages: git-clean html
	git co gh-pages
	git rm -r .
	git checkout HEAD -- .gitignore README.md .nojekyll
	cp -r doc/_build/html/* . # your sphinx build directory
	git stage .
	@echo 'Commit and push when ready or git reset --hard && git checkout master to revert'
