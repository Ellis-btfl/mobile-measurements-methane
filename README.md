# mobile-measurements-methane
Adapted algorithm from Tettenborn et al. (2025) for CH4 leaks quantification + chinese postman for survey routes

Cities_PeakProcessing_adapted, Methane_Mapping_Cities_modif and all the small functions they use come from Judith Tettenborn https://github.com/judith-tettenborn/CRE_CH4Quantification/tree/main and Roberto Paglini. I've only made some minor adjustments.

These algorithms require the analyser’s data files and the corresponding GPX tracks for these measurements as input. Processing_merging uses these files to create the merged CSV files that are subsequently used.

Methane_Mapping__Cities_modif uses the CSV files obtained and searches for recorded methane peaks. This code then generates a Mysurvey.csv file containing the characteristics of all identified peaks and, for each measurement session, produces an HTML graph showing the raw methane curve, the smoothed curve, the detection threshold and the area of the peaks.

Cities_PeakProcessing_adapted carries out the clustering. It reads the MySurvey.csv file and groups geographically close peaks into clusters. This results in the creation of a GPKG file containing the methane peaks grouped into clusters, along with a centroid for each cluster, as well as the creation of an HTML map allowing for the quick visualisation of the peaks grouped into clusters and the isolated peaks.

Corrélation_éthane uses the GPKG file containing the clusters, as well as the original data files from the analyser, to determine whether the selected methane peaks correlate with a simultaneous ethane peak

postier_chinois_polygone aims to generate routes to be followed during mobile survey sessions. It takes as input the latitude, longitude and altitude coordinates of the vertices of the polygon corresponding to the neighbourhood to be surveyed, and determines the shortest route that passes through every street (excluding dead ends) in the connected area at least once by resolving the chinese postman problem. The other version allows you to identify which streets correspond to gas mains, so that the survey can be focused on these streets.  It is recommended that you do not use the network’s GPKG file if, for the selected area, it is incomplete  and/or highly fragmented. 
