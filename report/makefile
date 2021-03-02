text_sources = $(addpostfix content $(addpostfix .tex main.tex))
figs = $(addprefix figs/, )
latex_temps = main.aux main.bbl main.blg main.log main.toc main.out \
main.fls main.xdv missfont.log main.fdb_latexmk main.run.xml main.bcf "main.synctex(busy)"

pdf:  ${content} ${figs} References.bib
	pdflatex -halt-on-error main \
	&& bibtex main; \
	   biber main \
	&& pdflatex -halt-on-error main

open: pdf
	xdg-open main.pdf &

.PHONY : clean
clean :
	-rm ${latex_temps};
	-rm main.pdf
