import os
from PyPDF2 import PdfFileMerger
from sys import argv

merger = PdfFileMerger()

for i in range(1, 109):
	merger.append(f"SKL_BDT-out{i}_output.pdf")

merger.write("../output.pdf")