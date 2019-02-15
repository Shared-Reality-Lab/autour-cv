Based on notes from Ani Dever, who was working on the OCR pipeline.

similarity.cpp looks for "similarity of two images" in case when we want to compare the freshly taken shot with the last uploaded one. There are 5 different methods, all implemented OpenCV itself. I have seen that brute-force and FLANN matching takes more than few seconds. I think PSNR, Histogram matching and MSSIM can provide good metrics. I have included a link at the beginning of each function, there is a description of theory and sample limits (e.g. for PSNR if the images significantly differ you'll get much lower ones like 15 and so).

blur.cpp evaluates the focus of the image with different methods, I have included the links as well. There is no detailed description, so I had to play with the methods and find out manually. For values below ~0.16, ~12, ~74, ~740 (respectively: canny edges, modifiedLaplacian, varianceOfLaplacian, normalizedGraylevelVariance. and greater values are better),
