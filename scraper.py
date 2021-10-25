#Import BeautifulSoup
from bs4 import BeautifulSoup
#import requests | Request Info from specific website like a real person
import requests
#Import NLTK toolkit
import nltk
from nltk import *
from nltk.corpus import stopwords
from nltk.text import Text
from nltk.collocations import *
import math

#start program
print("\n\n---- Web Scraper ----\n\n")
size = int(input('Inserte número de documentos para analizar: '))  #numero total de archivos


def scraper(size):
    global keywords_lists_normalized
    contador = 0
    #This is the number of papers the repositry display per request.
    total_documentos= 20
    #It will start counting from offset='0' to the specified number of documents
    pagina = "https://repository.javeriana.edu.co/handle/10554/9062/recent-submissions?offset="
    cont2=0

    #Here the values scraped are stored as lists
    titulos_list =[]
    autores_list = []
    dates_list = []
    abstact_list = []
    keywords_lists = []

    #The loop will stop when the size of documents reaches the target
    while contador<=size:

        if total_documentos <= size:
            html_text = requests.get(f'{pagina}{contador}').text
            soup = BeautifulSoup(html_text,'lxml')
            titulos_tags = soup.find_all('div', class_='artifact-description')

            for titulos in (titulos_tags):
                #This gets the handle id
                all_a = soup.select('.artifact-title a')
                for a in all_a:
                    handle_id= a['href']
                    handle_id_complete = "https://repository.javeriana.edu.co/"+handle_id
                    tesis_page = handle_id_complete
                    html_text_tesis_page = requests.get(f'{tesis_page}').text
                    soup = BeautifulSoup(html_text_tesis_page,'lxml')
                    titulo = soup.find('h2', class_='page-header').text
                    #print(titulo)
                    titulos_list.append(titulo)

                    #Getting the authors
                    if (soup.find('div', class_='simple-item-view-authors')) is not None:
                        autor_soap = soup.find('div', class_='simple-item-view-authors')
                        autor_scrapped= autor_soap.a.text
                        #print(autor_scrapped)
                        autores_list.append(autor_scrapped)
                    else:
                        autor_scrapped= "NO-AUTOR"
                        autores_list.append(autor_scrapped)
                        pass
                    #Getting the year
                    if (soup.find('div', class_='simple-item-view-date')) is not None:
                        year_soap  = soup.find('div', class_='simple-item-view-date').text
                        year_scrapped = year_soap.replace("\nDate","")
                        #print(year_scrapped)
                        dates_list.append(year_scrapped)
                    else:
                        year_scrapped= "NO-YEAR"
                        dates_list.append(autor_scrapped)
                        pass
                    #Getting the abstract
                    #If theres no abstract print error and continues
                    if (soup.find('div', class_='simple-item-view-description')) is not None:
                        abstract_scrapped = soup.find('div', class_='simple-item-view-description').text
                        abstract_scrapped_txt = abstract_scrapped.replace("\nResumen","")
                        #print(abstract_scrapped_txt)
                        abstact_list.append(abstract_scrapped_txt)
                    else:
                        #print("Sorry there was an error. This article does not contain and abstract")
                        abstract_scrapped_txt = "NO-RESUMEN"
                        abstact_list.append(abstract_scrapped_txt)
                        pass

                    #Getting the Keywords
                    pattern = "/browse?type=subject&amp;value"
                    #keyword_soap  = soup.find_all('a', {'href=/browse?type=subject&value=': True})
                    substring = '/browse?type=subject&value='
                    for keyword in soup.select('.simple-item-view-authors a'):
                        adjusted_keyword = keyword['href']
                        #print(adjusted_keyword)
                        if adjusted_keyword.find(substring)== -1:
                            #print("not found")
                            pass
                        else:
                            #print("***SI ES LO QUE BUSCO***")
                            keyword_scrapped_txt = keyword.text
                            #print(keyword_scrapped_txt)
                            keywords_lists.append(keyword_scrapped_txt)
                            #with open("keywords_output2.txt","a",encoding='utf8') as f:
                            #    print(f'{keyword_scrapped_txt}',file=f) #print all keywords


                    with open("output.txt","a",encoding='utf8') as f:

                        #title
                        titulo_clean = str(titulo)
                        id_titulo_clean = titulo_clean.replace("\t", "").replace("\r", "").replace("\n", "")

                        #author
                        author_clean = str(autor_scrapped)
                        id_author_clean = author_clean.replace("\t", "").replace("\r", "").replace("\n", "")

                        #abstract
                        abs_clean = str(abstract_scrapped_txt)
                        id_abs_clean = abs_clean.replace("\t", "").replace("\r", "").replace("\n", "")

                        #date
                        year_clean = str(year_scrapped)
                        id_year_clean = year_clean.replace("\t", "").replace("\r", "").replace("\n", "")

                        #Keyword list
                        keywords_clean = str(keyword_scrapped_txt)
                        id_keywords_clean = keywords_clean.replace("\t", "").replace("\r", "").replace("\n", "")
                        keywords_list_clean = str(keywords_lists[:])
                        id_keywords_list_clean = keywords_list_clean.replace("\t", "").replace("\r", "").replace("\n", "")
                        #add to document
                        print(f'{id_titulo_clean}|{id_titulo_clean}{id_abs_clean}{id_keywords_list_clean}|{id_year_clean}',file=f)
                    #clean list of keywords for each document
                    keywords_lists.clear()
            total_documentos+=20
            contador+= 20
            cont2+=1

        else:
            print("Se ha ejecutado el programa.")
            break

#Ejectuar funcion

if size >19:
    print("Se estan analizando los documentos. Esto puede tardar un poco.")
    scraper(size)
else:
    print("El programa funciona con minimo 20 documentos.")
