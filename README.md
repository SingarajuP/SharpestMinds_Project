# SharpestMinds_Project

## Overview
Reading books is a recreational activity. Based on the reviews for the selected book this app will show the percentage of emotions experienced by the customers who read the same book. 


## Background and Motivation

The reader will go through different emotions while reading a book. The feedback given as a review is based on the emotions they feel. These reviews are helpful for a user who wants to read a book. By analyzing the reviews given by different customers, we want to capture the emotions triggered for the book and the percentage of emotions while the customers who read it before experienced. 

While there is no data for books with labels for the emotions, the approach we plan is to create a model for the labeled text data which is available from the huggingface/kaggle websites. 

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
