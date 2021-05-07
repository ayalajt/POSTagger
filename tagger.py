#############################################################################################################################
# File:        tagger.py
# Author:      Jesus Ayala
# Date:        03/15/2021
# Class:       CMSC 416
# Description: A POS Tagger that tags a text file given a training file 
#############################################################################################################################

import sys
import re
import collections
from collections import Counter

# POS tagging refers to assigning a class tag to a word from a text file. For this implementation, it takes a text file that is used to train the tagger 
# with what tags are with which word. From here, it then takes a test file to be tagged and tags the words accordingly, using the data from the training file. 
# For example, if the training file had the lines "moderate/JJ allies/NNS", the tagger would save the word moderate with the adjective tag (JJ) and the word allies 
# with the plural noun tag (NNS). Then, if these words popped up in the test file, it would tag the words with those same tags. My train of thought was to first simply 
# go through the training file, storing every word with a single tag in a dictionary. If the word had multiple tags, then it would only accept the first tag. Then, it 
# would iterate through the file to be tagged, going word per word. For each word, if it is in the dictionary, assign it the tag the stored word has, and then move on to 
# the next word. If the word does not exist in the dictionary, assign it the noun tag (NN). This worked well, as according to my scorer, it led to an overall accuracy 
# score of 82.96% when compared to the  test key. I then decided to add a few rules. 
# Rule 1: If the word is purely digits, assign it the number tag (CD). This led to a small increase in the accuracy score, as it jumped to an overall accuracy score of 83.15%. 
# Rule 2: If the word ends with "s" or "es", then assign it the plural noun  tag (NNS) instead of just the noun tag. This increased the overall accuracy to 84.36%. 
# Rule 3: If the word ends with "ing", assign it the present verb tag (VBG). This led to a small increase to 84.64%. Rule 4: If the word starts with a capital letter, 
# assign it the proper noun tag (NNP), as it usually is a person or a place. This saw the biggest increase to the score, as it jumped to 88.12%. 
# Rule 1.5: I edited rule 1 a bit at this point, making the number tag much more aggresive, as it now is assigned to any word that contains at least one digit. This led to a 
# small increase to 88.63%. 
# Rule 4: If the word ends with "ive", assign it the adjective tag (JJ). Once again, it led to a small increase in the score to 88.66%.
# Rule 5: If the word ends with "ly", assign it the adverb tag (RB). Another small increase resulted from this, as it changed the score to 88.8%.
# After this, I decided to configure the most likely baseline and store all the tags from a word in a list. These tags are then counted, and the word is given the most
# frequent tag. This meant that if a word had multiple tags from the training file, the word is only given the tag that has the highest frequency. This led to a higher accuracy
# score of 90.23% once implemented. From here, I decided to implement a bigram model to predict the next tag given the previous tag. I used the same process from programming
# assignment 2, however, it led to a lower accuracy score (it went from 90.23% to 86.94%). As such, I left the code in the file but it is commented out.
# For example input, the file can be run from the command line and it needs two text files: the training file and a testing file. It can then be printed to another file with
# the command: "python tagger.py pos-train.txt pos-test.txt > taggedFile.txt". Then output will then be printed onto the taggedFile text file, where each word has a tag after it.

def main():

    # The first part of this program grabs the name of the two files to be used
    fileForTraining = -1
    fileToTag = -1
    try:
        fileForTraining = sys.argv[1]
        fileToTag = sys.argv[2]
    except:
        print("More arguments required")
        quit()

    openTrainingFile = open(fileForTraining, encoding = "utf8")
    line = ""
    tag = ""
    word = ""
    tagFreqDictionary = {}
    wordTagDictionary = {}
    tagsTogether = ""

    # Go through each line in the training file, ignore brackets
    for line in openTrainingFile:
        line = line.replace('[', "").replace(']',"")
        line = line.split()

        # go through each word in the line, splitting the word up by the / character, and grabbing the tag after it
        for i in line:

            # if the word contains a backslash then it is escaping the / character, so grab the value after the next / character
            if "\\" in i:
                tag = i.split("/")
                word = tag[0] + "/" + tag[1]
                tag = tag[2]
            else:
                tag = i.split("/")

                # if the word has two tags, choose the first tag
                if "|" in tag[1]:
                    tag = tag[1]
                    tag = tag.split("|")
                    tag = tag[0]
                else:

                    # otherwise just grab the tag after the / character
                    word = tag[0]
                    tag = tag[1]

            # store the tags in a frequency dictionary
            if tag in tagFreqDictionary:
                tagFreqDictionary[tag] = tagFreqDictionary.get(tag) + 1
            else:
                tagFreqDictionary[tag] = 1

            # Keep track of each word's tag and store it in another dictionary
            if word not in wordTagDictionary:
                # TODO: make it hold multiple values and its frequency (that way the most frequent tag can be chosen if the word exists)
                wordTagDictionary[word] = [tag]
            # word is in dictionary, update frequency
            else:
                wordTagDictionary[word].append(tag)

            # Also store the tags in a giant string in order to create a bigram later
            tagsTogether = tagsTogether + " " + tag


    # This new code finds the most frequent tag given a word's tag list and stores the tag with the word
    newWordTagDict = {}
    for key in wordTagDictionary:
        values = wordTagDictionary.get(key)
        count = Counter()
        for word in values:
            count[word] += 1
            findTag = count.most_common(1)
            findTag = findTag[0][0]
        newWordTagDict[key] = findTag

    # Creating bigram table
    # Create a table for the bigrams
    bigramList = createBigrams(tagsTogether)
    bigramTable = Counter()
    bigramTable.update(bigramList)
 
    # Create a bigram probability table, which stores the probability of each bigram by dividing the frequency of the bigram by the frequency of the first word
    # (Implemented the same as from ngram.py)
    bigramProbTable = []
    for val in bigramTable:
        theTags = val.split(" ")
        firstTag = theTags[0]
        firstTagFreq = tagFreqDictionary.get(firstTag)
        if firstTagFreq is not None:
            prob = bigramTable[val] / firstTagFreq
            prob = round(prob, 3)
            bigramProbTable.append((val, prob))
    #print(bigramProbTable)
    openFileToTagFile = open(fileToTag, encoding = "utf8")
    wordTag = ""

    # Tagging the file
    for line in openFileToTagFile:
        splitLine = line.split()

        # Go through the line of the to be tagged file, grabbing each word and assigning it a tag
        for j in range(len(splitLine)):
            currentWord = splitLine[j]
            if currentWord == "[" or currentWord == "]":
                pass
            else:

                # if the word is found in the dictionary, then grab the tag and add it to the word
                if currentWord in newWordTagDict:
                    wordTag = newWordTagDict.get(currentWord)
                    wordTag = wordTag

                    # replace element in list with word + tag, print listas a line at the end
                    wordWithTag = currentWord + "/" + wordTag
                    splitLine[j] = wordWithTag
                

                else:

                    # code to choose a tag for an ambigious word given the previous tag
                    #prevWord = splitLine[j-1]
                    #prevWordTag = prevWord.split("/")
                    #if (len(prevWordTag) > 1):
                        #prevWordTag = prevWord[1]
                        #possibleTagsProbabilityList = []
                        #for currentWords in bigramProbTable:
                            #totalWords = currentWords[0]
                            #possibleWords = totalWords.split(" ")
                            #if prevWordTag == possibleWords[0]:
                                #possibleTagsProbabilityList.append(currentWords)
                        #possibleTagsProbabilityList.sort(key=lambda x:x[1])
                        #if (len(possibleTagsProbabilityList) > 1):
                            #tagChosen = possibleTagsProbabilityList[0]
                            #tagChosen = tagChosen[0].split(" ")
                            #tagChosen = tagChosen[1]
                            #wordWithTag = currentWord + "/" + tagChosen

                    # if word is purely numbers, assign it number tag
                    if currentWord.isdigit():
                        wordWithTag = currentWord + "/CD"
                    
                    # if the word is an equal sign, assign it symbol tag
                    elif currentWord == "=":
                        wordWithTag = currentWord + "/SYM"
                    
                    # if word ends with es or s, assign it as a plural noun 
                    elif currentWord.endswith("es") or currentWord.endswith("s"):
                        wordWithTag = currentWord + "/NNS"
                    
                    # if word ends with ing, assign it the present verb tag
                    elif currentWord.endswith("ing"):
                        wordWithTag = currentWord + "/VBG"
                    
                    # if the word is capitiliazed, assign it the proper noun tag
                    elif currentWord[0].isupper():
                        wordWithTag = currentWord + "/NNP"
                    
                    # if the word ends with ive, assign it adjective tag
                    elif currentWord.endswith("ive"):
                        wordWithTag = currentWord + "/JJ"
                    
                    # if the word contains any numbers, assign it number tag
                    elif containsDigit(currentWord):
                        wordWithTag = currentWord + "/CD"
                    
                    # if the word ends with y, assign it the adverb tag
                    elif currentWord.endswith("ly"):
                        wordWithTag = currentWord + "/RB"
                    
                    # otherwise, just give the word the noun tag
                    else:
                        wordWithTag = currentWord + "/NN"
                    splitLine[j] = wordWithTag

        # print each line in the ouput text file, cleaning up any spaces
        print(" ".join(splitLine))


def createBigrams(stringOfTags):

    # To start tokenizing the string, first split it up by the spaces
    tokens = stringOfTags.split(" ")

    # Then use zip, which was grabbed from http://www.locallyoptimal.com/blog/2013/01/20/elegant-n-gram-generation-in-python/. Here, it passes
    # the a list with *, where it would be tokens, tokens[1:], tokens[2:], and so on until it hits the number of ngrams needed. It then creates 
    # the ngram as zip takes a value from each list and staggers them. It then combines them all in a list and returns this list
    findNGrams = zip(*[tokens[i:] for i in range(2)])
    aList = [" ".join(currentNGram) for currentNGram in findNGrams] 
    return aList

# helper function to check if there exists at least one digit in a string
def containsDigit(word):
    for char in word:
        if char.isdigit():
            return True
    return False

if __name__ == '__main__':
    main()