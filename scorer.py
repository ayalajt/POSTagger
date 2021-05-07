#############################################################################################################################
# File:        scorer.py
# Author:      Jesus Ayala
# Date:        03/15/2021
# Class:       CMSC 416
# Description: A scoring file from a tagger; it takes two files and compares the accuracy between the two
#############################################################################################################################

import sys
import pandas as pd

# The scorer file compares two files, a file that is tagged using a tagger, and a file that is the tagged file key, and compares the
# similarity between the two's tags. This results in an accuracy score being calculated as well as a confusion matrix. This is done by going 
# word for word in each file, grabbing the tag, storing them in a string, and then once this is finished, compare the tags at the same part of the string. 
# If they are the same, then it was tagged correctly. Then divide the number of correctly tagged words by the total number of tags, which results in the accuracy 
# score being printed. For the confusion matrix, pandas was used by running "pip install pandas" in the command line. It is then passed a list containing all the 
# training file tags as well as a row that contains all the key file tags. It then uses these together to create the confusion matrix. An example input would be 
# "python scorer.py taggedFile.txt pos-test-key.txt > results.txt", and the output will contain the accuracy score at the top and the confusion matrix down below.


def main():

    # First grab the file names
    fileTagged = -1
    fileTaggedKey = -1
    try:
        fileTagged = sys.argv[1]
        fileTaggedKey = sys.argv[2]
    except:
        print("More arguments required")
        quit()


    tagStringTrained = ""
    tagStringKey = ""

    # For the tagged file, go through the file and grab every tag for a word and store them in a string
    openTaggedFile = open(fileTagged, encoding = "utf8")
    for lineOne in openTaggedFile:
        lineOne = lineOne.replace('[', "").replace(']',"")
        lineOne = lineOne.split()
        for wordTag in lineOne:
            if "\\" in wordTag:
                tag = wordTag.split("/")
                tag = tag[2]
                tagStringTrained = tagStringTrained + tag + " "
            else:
                tag = wordTag.split("/")
                tag = tag[1]
                tagStringTrained = tagStringTrained + tag + " "
        openTaggedFile = open(fileTagged, encoding = "utf8")

    # For the tag key file, go through the file and grab every tag for a word and store them in a string 
    openKeyFile = open(fileTaggedKey, encoding = "utf8")
    for lineOne in openKeyFile:
        lineOne = lineOne.replace('[', "").replace(']',"")
        lineOne = lineOne.split()
        for wordTag in lineOne:
            if "\\" in wordTag:
                tag = wordTag.split("/")
                tag = tag[2]
                tagStringKey = tagStringKey + tag + " "
            else:
                tag = wordTag.split("/")
                tag = tag[1]
                tagStringKey = tagStringKey + tag + " "
    tagFile = tagStringTrained.split()
    tagKey = tagStringKey.split()

    sameTags = 0
    tagTotal = len(tagFile)

    # iterate through the number of total tags (which would be the same total for both files) and compare the tags at the same index.
    # if they are the same, then it was tagged correctly, so increment the same tags value
    for i in range(tagTotal):
        if tagFile[i] == tagKey[i]:
            sameTags = sameTags + 1
    
    # Calculate the accuracy by dividing the number of correct tags by the total tags, multiplying by 100, and then rounding to 2 significant digits
    accuracy = sameTags / tagTotal
    accuracy = accuracy * 100
    accuracy = round(accuracy, 2)
    print("Overall Accuracy: " + str(accuracy) + "%")
    print()

    # Here, pandas is used for the confusion matrix. The first four lines are formatting options, and then actual contains the row with the key file's tags
    # and predicted contains the trained file's tags. Create a matrix using both of these and print the results
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    actual = pd.Series(tagKey, name="Actual")
    predicted = pd.Series(tagFile, name="Predicted")
    matrix = pd.crosstab(actual, predicted)
    print(matrix)
            
if __name__ == '__main__':
    main()