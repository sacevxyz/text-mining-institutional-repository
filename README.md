# Text mining institutional repositories using Python

Text mining script to extract data from Dspace repositories.


This tool allows you to extract metadata from Dspace-based repositories.  It can obtain the author, titles, abstracts, keywords and much more from documents (theses).

It contains two scripts. The **scrapper.py** script downloads the metadata to a local drive. Once executed it creates a .txt file as output. Run **data_analysis.py** to read the .txt file and find trends or predefined categories that are contained in the /data folder.  It makes use of TF-IDF and cosine similarity techniques to sort the documents with more similarity according to a specific topic.
