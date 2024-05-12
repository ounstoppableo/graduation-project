from __future__ import division 
import csv
import os
from packages.getNodeMetric import util
from packages.getNodeMetric import astChecker
from packages.getNodeMetric import customast
import json
import math
import pyan


def run(target_dir,output_dir):
	ClassFeatureCsvFile = open(f'{output_dir}\\class.csv', 'w', newline='')
	ClassFeature = csv.writer(ClassFeatureCsvFile)
	ClassFeature.writerow(['type','className','fileName','lineno','CLOC','NOA','NOM','RCLOC'])
	DefFeatureCsvFile = open(f'{output_dir}\\function.csv', 'w', newline='')
	DefFeature = csv.writer(DefFeatureCsvFile)
	DefFeature.writerow(['type','defName','fileName','lineno','MLOC','PAR','DOC','RMLOC'])

	for currentFileName in util.walkDirectory(target_dir):
		try:
			astContent = customast.parse_file(currentFileName)
		except Exception as e:
			print (target_dir,currentFileName)
			continue
		myast = astChecker.MyAst()
		myast.fileName = currentFileName
		myast.visit(astContent)
		featureRow = {}
		for item in myast.result:
			if item[1]=='DEF':
				try:
					featureRow[item[3]] 
				except:
					featureRow[item[3]] = {}
				try:
					featureRow[item[3]][item[2]] 
				except:
					featureRow[item[3]][item[2]] = {'lineno':item[4],'type':'DEF','MLOC':0,'PAR':0,'DOC':0,'RMLOC':0}
				if item[0] == 'MLOC':
					featureRow[item[3]][item[2]]['MLOC'] = item[5]
				if item[0] == 'PAR':
					featureRow[item[3]][item[2]]['PAR'] = item[5]
				if item[0] == 'DOC':
					featureRow[item[3]][item[2]]['DOC'] = item[5]
				if item[0] == 'RMLOC':
					featureRow[item[3]][item[2]]['RMLOC'] = item[5]
			elif item[1]=='CLASS':	
				try:
					featureRow[item[3]] 
				except:
					featureRow[item[3]] = {}
				try:
					featureRow[item[3]][item[2]] 
				except:
					featureRow[item[3]][item[2]] = {'lineno':item[4],'type':'CLASS','CLOC':0,'NOA':0,'NOM':0,'RCLOC':0}
				if item[0] == 'CLOC':
					featureRow[item[3]][item[2]]['CLOC'] = item[5]
				if item[0] == 'NOA':
					featureRow[item[3]][item[2]]['NOA'] = item[5]
				if item[0] == 'NOM':
					featureRow[item[3]][item[2]]['NOM'] = item[5]
				if item[0] == 'RCLOC':
					featureRow[item[3]][item[2]]['RCLOC'] = item[5]
		for key in featureRow:
			for key2 in featureRow[key]:
				if featureRow[key][key2]['type'] == 'DEF':
					DefFeature.writerow([featureRow[key][key2]['type'],key2,key,featureRow[key][key2]['lineno'],featureRow[key][key2]['MLOC'],featureRow[key][key2]['PAR'],featureRow[key][key2]['DOC'],featureRow[key][key2]['RMLOC']])
				elif featureRow[key][key2]['type'] == 'CLASS':
					ClassFeature.writerow([featureRow[key][key2]['type'],key2,key,featureRow[key][key2]['lineno'],featureRow[key][key2]['CLOC'],featureRow[key][key2]['NOA'],featureRow[key][key2]['NOM'],featureRow[key][key2]['RCLOC']])

	ClassFeatureCsvFile.close()
	DefFeatureCsvFile.close()