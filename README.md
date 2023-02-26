# SharpestMinds_Project

## Overview
Reading books and watching movies is a recreational activity. Based on the reviews for the books this app will show the percentage of customers who felt the same emotions while reading the same book. For movies, this app will show the percentage of different emotions with start and stop time for the emotion based on the subtitles of the movies. 


## Background and Motivation

The reader will go through different emotions while reading a book. The feedback given as a review is based on the emotions they feel. These reviews are helpful for a user who wants to read a book. By analyzing the reviews given by different customers, we want to capture the emotions triggered for the book and the percentage of people who felt the same emotion. 
In movies, the dialogues produce different emotions to the viewer. From the dialogues we want to capture the percentage of emotions in the whole movie. 
While there is no data for individual books or movie subtitles with labels for the emotions, the approach we plan is to create a model for the labeled text data which is available from the huggingface/kaggle websites. 

## Goals

## Datasets

## Usage
Clone repo 
```bash
 git clone https://github.com/SingarajuP/sharpestminds-project.git
```
<br />Setup a virtual environment
```bash
conda create -n yourenvname python=3.10.4
```
<br />Activate the virtual environment

```bash
conda activate yourenvname
```
<br />Install all requirements using pip:
```bash
pip install -r requirements.txt
```
<br />To run web application stay in the main directory and run the command:
```bash
uvicorn main:app --reload 
After this you get Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
open http://127.0.0.1:8000/docs in the browser
```



## Practical Applications

## Milestones

## References
