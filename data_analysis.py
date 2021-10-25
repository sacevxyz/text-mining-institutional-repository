import nltk
from nltk import *
from nltk.corpus import stopwords
from nltk.text import Text
from nltk.collocations import *
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import SnowballStemmer
import csv
import pandas as pd

#Categorias
raw_comunicacion_organizacional = []
raw_creacion_y_produccion_audiovisual = []
raw_creacion_y_produccion_editorial_y_multimedial = []
raw_creacion_y_produccion_sonora_y_radiofonica = []
raw_periodismo = []
raw_publicidad = []

with open("data/categorias/periodismo.txt", "r",encoding='utf8') as file_1:
  for line in file_1:
    raw_periodismo.append(line.strip())

with open("data/categorias/publicidad.txt", "r",encoding='utf8') as file_2:
  for line in file_2:
    raw_publicidad.append(line.strip())

with open("data/categorias/comunicacion_organizacional.txt", "r",encoding='utf8') as file_3:
  for line in file_3:
    raw_comunicacion_organizacional.append(line.strip())

with open("data/categorias/creacion_y_produccion_audiovisual.txt", "r",encoding='utf8') as file_4:
  for line in file_4:
    raw_creacion_y_produccion_audiovisual.append(line.strip())

with open("data/categorias/creacion_y_produccion_editorial_y_multimedial.txt", "r",encoding='utf8') as file_5:
  for line in file_5:
    raw_creacion_y_produccion_editorial_y_multimedial.append(line.strip())

with open("data/categorias/creacion_y_produccion_sonora_y_radiofonica.txt", "r",encoding='utf8') as file_6:
  for line in file_6:
    raw_creacion_y_produccion_sonora_y_radiofonica.append(line.strip())



file_1.close()
file_2.close()
file_3.close()
file_3.close()
file_3.close()
file_3.close()


#Tendencias
raw_activismo_laboral = []
raw_adultez_emergente = []
raw_anonimato = []
raw_aprendizaje_conectado = []
raw_aprendizaje_invertido = []
raw_avances_en_el_envejecimiento = []
raw_blockchain = []
raw_cajas_de_suscripcion = []
raw_cambio_de_privacidad = []
raw_ciudades_inteligentes = []
raw_coches_autonomos = []
raw_control_por_voz = []
raw_coworking = []
raw_datos_en_todas_partes = []
raw_desenchufado = []
raw_desigualdad_de_ingresos = []
raw_drones = []
raw_economia_colaborativa = []
raw_fandom = []
raw_fast_casual = []
raw_gamificacion = []
raw_impacto_colectivo = []
raw_influencia_de_la_empresa = []
raw_insignias = []
raw_inteligencia_artificial = []
raw_internet_de_las_cosas = []
raw_juguetes_conectados = []
raw_lectura_corta = []
raw_lugares_creativos = []
raw_micromovilidad = []
raw_movimiento_maker = []
raw_nativos_digitales = []
raw_procesamiento_de_diseno = []
raw_realidad_virtual = []
raw_reconocimiento_facial = []
raw_renta_basica = []
raw_repensar_lo_rural = []
raw_resiliencia = []
raw_robots = []
raw_tecnologia_haptica = []
raw_urbanizacion = []
raw_venta_al_por_menor_experiencial = []

with open('data/tendencias/activismo_laboral.txt', 'r',encoding='utf-8') as tend_file_1:
    for line in tend_file_1:
        raw_activismo_laboral.append(line.strip())

with open('data/tendencias/adultez_emergente.txt', 'r',encoding='utf-8') as tend_file_2:
    for line in tend_file_2:
        raw_adultez_emergente.append(line.strip())

with open('data/tendencias/anonimato.txt', 'r',encoding='utf-8') as tend_file_3:
    for line in tend_file_3:
        raw_anonimato.append(line.strip())

with open('data/tendencias/aprendizaje_conectado.txt', 'r',encoding='utf-8') as tend_file_4:
    for line in tend_file_4:
        raw_aprendizaje_conectado.append(line.strip())

with open('data/tendencias/aprendizaje_invertido.txt', 'r',encoding='utf-8') as tend_file_5:
    for line in tend_file_5:
        raw_aprendizaje_invertido.append(line.strip())

with open('data/tendencias/avances_en_el_envejecimiento.txt', 'r',encoding='utf-8') as tend_file_6:
    for line in tend_file_6:
        raw_avances_en_el_envejecimiento.append(line.strip())

with open('data/tendencias/blockchain.txt', 'r',encoding='utf-8') as tend_file_7:
    for line in tend_file_7:
        raw_blockchain.append(line.strip())

with open('data/tendencias/cajas_de_suscripcion.txt', 'r',encoding='utf-8') as tend_file_8:
    for line in tend_file_8:
        raw_cajas_de_suscripcion.append(line.strip())

with open('data/tendencias/cambio_de_privacidad.txt', 'r',encoding='utf-8') as tend_file_9:
    for line in tend_file_9:
        raw_cambio_de_privacidad.append(line.strip())

with open('data/tendencias/ciudades_inteligentes.txt', 'r',encoding='utf-8') as tend_file_10:
    for line in tend_file_10:
        raw_ciudades_inteligentes.append(line.strip())

with open('data/tendencias/coches_autonomos.txt', 'r',encoding='utf-8') as tend_file_11:
    for line in tend_file_11:
        raw_coches_autonomos.append(line.strip())

with open('data/tendencias/control_por_voz.txt', 'r',encoding='utf-8') as tend_file_12:
    for line in tend_file_12:
        raw_control_por_voz.append(line.strip())

with open('data/tendencias/coworking.txt', 'r',encoding='utf-8') as tend_file_13:
    for line in tend_file_13:
        raw_coworking.append(line.strip())

with open('data/tendencias/datos_en_todas_partes.txt', 'r',encoding='utf-8') as tend_file_14:
    for line in tend_file_14:
        raw_datos_en_todas_partes.append(line.strip())

with open('data/tendencias/desenchufado.txt', 'r',encoding='utf-8') as tend_file_15:
    for line in tend_file_15:
        raw_desenchufado.append(line.strip())

with open('data/tendencias/desigualdad_de_ingresos.txt', 'r',encoding='utf-8') as tend_file_16:
    for line in tend_file_16:
        raw_desigualdad_de_ingresos.append(line.strip())

with open('data/tendencias/drones.txt', 'r',encoding='utf-8') as tend_file_17:
    for line in tend_file_17:
        raw_drones.append(line.strip())

with open('data/tendencias/economia_colaborativa.txt', 'r',encoding='utf-8') as tend_file_18:
    for line in tend_file_18:
        raw_economia_colaborativa.append(line.strip())

with open('data/tendencias/fandom.txt', 'r',encoding='utf-8') as tend_file_19:
    for line in tend_file_19:
        raw_fandom.append(line.strip())

with open('data/tendencias/fast_casual.txt', 'r',encoding='utf-8') as tend_file_20:
    for line in tend_file_20:
        raw_fast_casual.append(line.strip())

with open('data/tendencias/gamificacion.txt', 'r',encoding='utf-8') as tend_file_21:
    for line in tend_file_21:
        raw_gamificacion.append(line.strip())

with open('data/tendencias/impacto_colectivo.txt', 'r',encoding='utf-8') as tend_file_22:
    for line in tend_file_22:
        raw_impacto_colectivo.append(line.strip())

with open('data/tendencias/influencia_de_la_empresa.txt', 'r',encoding='utf-8') as tend_file_23:
    for line in tend_file_23:
        raw_influencia_de_la_empresa.append(line.strip())

with open('data/tendencias/insignias.txt', 'r',encoding='utf-8') as tend_file_24:
    for line in tend_file_24:
        raw_insignias.append(line.strip())

with open('data/tendencias/inteligencia_artificial.txt', 'r',encoding='utf-8') as tend_file_25:
    for line in tend_file_25:
        raw_inteligencia_artificial.append(line.strip())

with open('data/tendencias/internet_de_las_cosas.txt', 'r',encoding='utf-8') as tend_file_26:
    for line in tend_file_26:
        raw_internet_de_las_cosas.append(line.strip())

with open('data/tendencias/juguetes_conectados.txt', 'r',encoding='utf-8') as tend_file_27:
    for line in tend_file_27:
        raw_juguetes_conectados.append(line.strip())

with open('data/tendencias/lectura_corta.txt', 'r',encoding='utf-8') as tend_file_28:
    for line in tend_file_28:
        raw_lectura_corta.append(line.strip())

with open('data/tendencias/lugares_creativos.txt', 'r',encoding='utf-8') as tend_file_29:
    for line in tend_file_29:
        raw_lugares_creativos.append(line.strip())

with open('data/tendencias/micromovilidad.txt', 'r',encoding='utf-8') as tend_file_30:
    for line in tend_file_30:
        raw_micromovilidad.append(line.strip())

with open('data/tendencias/movimiento_maker.txt', 'r',encoding='utf-8') as tend_file_31:
    for line in tend_file_31:
        raw_movimiento_maker.append(line.strip())

with open('data/tendencias/nativos_digitales.txt', 'r',encoding='utf-8') as tend_file_32:
    for line in tend_file_32:
        raw_nativos_digitales.append(line.strip())

with open('data/tendencias/procesamiento_de_diseno.txt', 'r',encoding='utf-8') as tend_file_33:
    for line in tend_file_33:
        raw_procesamiento_de_diseno.append(line.strip())

with open('data/tendencias/realidad_virtual.txt', 'r',encoding='utf-8') as tend_file_34:
    for line in tend_file_34:
        raw_realidad_virtual.append(line.strip())

with open('data/tendencias/reconocimiento_facial.txt', 'r',encoding='utf-8') as tend_file_35:
    for line in tend_file_35:
        raw_reconocimiento_facial.append(line.strip())

with open('data/tendencias/renta_basica.txt', 'r',encoding='utf-8') as tend_file_36:
    for line in tend_file_36:
        raw_renta_basica.append(line.strip())

with open('data/tendencias/repensar_lo_rural.txt', 'r',encoding='utf-8') as tend_file_37:
    for line in tend_file_37:
        raw_repensar_lo_rural.append(line.strip())

with open('data/tendencias/resiliencia.txt', 'r',encoding='utf-8') as tend_file_38:
    for line in tend_file_38:
        raw_resiliencia.append(line.strip())

with open('data/tendencias/robots.txt', 'r',encoding='utf-8') as tend_file_39:
    for line in tend_file_39:
        raw_robots.append(line.strip())

with open('data/tendencias/tecnologia_haptica.txt', 'r',encoding='utf-8') as tend_file_40:
    for line in tend_file_40:
        raw_tecnologia_haptica.append(line.strip())

with open('data/tendencias/urbanizacion.txt', 'r',encoding='utf-8') as tend_file_41:
    for line in tend_file_41:
        raw_urbanizacion.append(line.strip())

with open('data/tendencias/venta_al_por_menor_experiencial.txt', 'r',encoding='utf-8') as tend_file_42:
    for line in tend_file_42:
        raw_venta_al_por_menor_experiencial.append(line.strip())








#Cargar datos de tesis.
df = pd.read_csv('datos_limpios.csv')
datos_samples = df.sample(n = 2033, random_state=1)
datos_tesis = datos_samples.groupby(['title'])['text'].apply(','.join).reset_index()
#print(datos_tesis.shape)



#esto logra tokenizar los titulos
'''
df['tokenized_sents'] = df.apply(lambda row: nltk.word_tokenize(row['title']), axis=1)
titulos_tokenizados = (df['tokenized_sents'])
print(titulos_tokenizados)

df['tokenized_sents2'] = df.apply(lambda row: nltk.word_tokenize(row['text']), axis=1)
titulos_tokenizados2 = (df['tokenized_sents2'])
print(titulos_tokenizados2)

'''
#Transformar texto
def eliminarURLs(text):
  import re
  text = re.sub(r"http\S+", "", text)
  return text

def tokenizar(text):
  from nltk import TweetTokenizer
  tokenizerLocal = TweetTokenizer()
  tokens = tokenizerLocal.tokenize(text)
  return tokens

def eliminarMenciones(tokens):
  new_words = []
  for word in tokens:
      if "#" not in word:
        if "@" not in word:
          new_words.append(word)
  return new_words

def eliminarSignosPuntuacionTokens(tokens):
  import re,string
  new_words = []
  for word in tokens:
      new_word = re.sub('[%s]' % re.escape(string.punctuation), '', word)
      if new_word != '':
          new_words.append(new_word)
  return new_words

def limpiarStopwords(tokens):
  from nltk.corpus import stopwords
  stopwordsGenerales = stopwords.words('spanish')
  stopwordsLocales = ['¿','?','"','[',']','“']
  todosLosStopWords = stopwordsGenerales + stopwordsLocales
  #Selecciona las palabras sin los stopwords
  words_selected_with_stopwords = [w for w in tokens if w.lower() not in todosLosStopWords]
  return words_selected_with_stopwords

def eliminarEmoticon(text):
    return text.encode('ascii', 'ignore').decode('ascii')

def stemmerPersonal(text):
  from nltk.stem import SnowballStemmer
  stemmer_spanish = SnowballStemmer('spanish')
  return stemmer_spanish.stem(text)

def procesamiento(trinos):
    trinos_limpios = []
    for trino in trinos:
      texto_sin_url = eliminarURLs(trino)
      texto_tokenizado = tokenizar(texto_sin_url)
      texto_sin_menciones = eliminarMenciones(texto_tokenizado)
      texto_sin_signos_p = eliminarSignosPuntuacionTokens(texto_sin_menciones)
      texto_sin_stopwords = limpiarStopwords(texto_sin_signos_p)
      #texto_stem = stemmerPersonal(" ".join(texto_sin_stopwords))
      texto_sin_emoticones = eliminarEmoticon(" ".join(texto_sin_stopwords))
      trinos_limpios.append(texto_sin_emoticones)
    return trinos_limpios


#Categorias
periodismo = " ".join(procesamiento(raw_periodismo))
publicidad = " ".join(procesamiento(raw_publicidad))
comunicacion_organizacional = " ".join(procesamiento(raw_comunicacion_organizacional))
creacion_y_produccion_audiovisual = " ".join(procesamiento(raw_creacion_y_produccion_audiovisual))
creacion_y_produccion_editorial_y_multimedial = " ".join(procesamiento(raw_creacion_y_produccion_editorial_y_multimedial))
creacion_y_produccion_sonora_y_radiofonica = " ".join(procesamiento(raw_creacion_y_produccion_sonora_y_radiofonica))

#Tedencias
activismo_laboral = " ".join(procesamiento(raw_activismo_laboral ))
adultez_emergente = " ".join(procesamiento(raw_adultez_emergente ))
anonimato= " ".join(procesamiento(raw_anonimato ))
aprendizaje_conectado= " ".join(procesamiento(raw_aprendizaje_conectado ))
aprendizaje_invertido= " ".join(procesamiento(raw_aprendizaje_invertido ))
avances_en_el_envejecimiento= " ".join(procesamiento(raw_avances_en_el_envejecimiento ))
blockchain= " ".join(procesamiento(raw_blockchain ))
cajas_de_suscripcion= " ".join(procesamiento(raw_cajas_de_suscripcion ))
cambio_de_privacidad= " ".join(procesamiento(raw_cambio_de_privacidad ))
ciudades_inteligentes= " ".join(procesamiento(raw_ciudades_inteligentes ))
coches_autonomos= " ".join(procesamiento(raw_coches_autonomos ))
control_por_voz= " ".join(procesamiento(raw_control_por_voz ))
coworking= " ".join(procesamiento(raw_coworking ))
datos_en_todas_partes= " ".join(procesamiento(raw_datos_en_todas_partes ))
desenchufado= " ".join(procesamiento(raw_desenchufado ))
desigualdad_de_ingresos= " ".join(procesamiento(raw_desigualdad_de_ingresos ))
drones= " ".join(procesamiento(raw_drones ))
economia_colaborativa= " ".join(procesamiento(raw_economia_colaborativa ))
fandom= " ".join(procesamiento(raw_fandom ))
fast_casual= " ".join(procesamiento(raw_fast_casual ))
gamificacion= " ".join(procesamiento(raw_gamificacion ))
impacto_colectivo= " ".join(procesamiento(raw_impacto_colectivo ))
influencia_de_la_empresa= " ".join(procesamiento(raw_influencia_de_la_empresa ))
insignias= " ".join(procesamiento(raw_insignias ))
inteligencia_artificial= " ".join(procesamiento(raw_inteligencia_artificial ))
internet_de_las_cosas= " ".join(procesamiento(raw_internet_de_las_cosas ))
juguetes_conectados= " ".join(procesamiento(raw_juguetes_conectados ))
lectura_corta= " ".join(procesamiento(raw_lectura_corta ))
lugares_creativos= " ".join(procesamiento(raw_lugares_creativos ))
micromovilidad= " ".join(procesamiento(raw_micromovilidad ))
movimiento_maker= " ".join(procesamiento(raw_movimiento_maker ))
nativos_digitales= " ".join(procesamiento(raw_nativos_digitales ))
procesamiento_de_diseno= " ".join(procesamiento(raw_procesamiento_de_diseno ))
realidad_virtual= " ".join(procesamiento(raw_realidad_virtual ))
reconocimiento_facial= " ".join(procesamiento(raw_reconocimiento_facial ))
renta_basica= " ".join(procesamiento(raw_renta_basica ))
repensar_lo_rural= " ".join(procesamiento(raw_repensar_lo_rural ))
resiliencia= " ".join(procesamiento(raw_resiliencia ))
robots= " ".join(procesamiento(raw_robots ))
tecnologia_haptica= " ".join(procesamiento(raw_tecnologia_haptica ))
urbanizacion= " ".join(procesamiento(raw_urbanizacion ))
venta_al_por_menor_experiencial= " ".join(procesamiento(raw_venta_al_por_menor_experiencial ))



#SO FAR SO GOOD

def preprocesamiento_tesis_limpios(dataSet):
  for i, trial in dataSet.iterrows():
      dataSet.loc[i, "text"] = tokenizar(dataSet.loc[i, "text"])
      dataSet.loc[i, "text"] = eliminarSignosPuntuacionTokens(dataSet.loc[i, "text"])
      dataSet.loc[i, "text"] = limpiarStopwords(dataSet.loc[i, "text"])
  return dataSet


tesis_estudiantes_limpios = preprocesamiento_tesis_limpios(datos_tesis)

#print(type(tesis_estudiantes_limpios))
#print(tesis_estudiantes_limpios["text"].dtype)
tesis_estudiantes_limpios["text"] = tesis_estudiantes_limpios["text"].apply(', '.join)


# Build TF-IDF
tfidf = TfidfVectorizer()
tfs = tfidf.fit_transform(tesis_estudiantes_limpios["text"])


# Features
#print("n_docs: %d, n_features: %d" % tfs.shape)
docs_num, feature_num = tfs.shape
feature_names = tfidf.get_feature_names()
#print(feature_names)


#print("###### Calculo de Feature Names ######")
#for x in range(0, feature_num):
#    print(" # ", x ," - ",feature_names[x], " - ", [tfs[n,x] for n in range(0, docs_num)])


with open("output_features-5.txt","a",encoding='utf8') as f:
    for x in range(0, feature_num):
        print(x ,",",feature_names[x], ",", [tfs[n,x] for n in range(0, docs_num)],file=f)


'''Similaridad coseno'''

response = tfidf.transform([periodismo, publicidad, comunicacion_organizacional,creacion_y_produccion_audiovisual,creacion_y_produccion_editorial_y_multimedial,creacion_y_produccion_sonora_y_radiofonica,])
cosine_similarity_response =  cosine_similarity(response, tfs)
coseno = pd.DataFrame(cosine_similarity_response.T, index = tesis_estudiantes_limpios["title"], columns = ["periodismo", "publicidad","comunicacion_organizacional","creacion_y_produccion_audiovisual","creacion_y_produccion_editorial_y_multimedial","creacion_y_produccion_sonora_y_radiofonica"])

'''
#ind = np.arange[len(list_of_dfs)]
tesis_periodismo = coseno.sort_values(["periodismo"], ascending=False)
tesis_publicidad = coseno.sort_values(["publicidad"], ascending=False)
tesis_comunicacion_organizacional = coseno.sort_values(["comunicacion_organizacional"], ascending=False)
tesis_creacion_y_produccion_audiovisual= coseno.sort_values(["creacion_y_produccion_audiovisual"], ascending=False)
tesis_creacion_y_produccion_editorial_y_multimedial= coseno.sort_values(["creacion_y_produccion_editorial_y_multimedial"], ascending=False)
tesis_creacion_y_produccion_sonora_y_radiofonica= coseno.sort_values(["creacion_y_produccion_sonora_y_radiofonica"], ascending=False)
'''

coseno.to_csv('similitud_tesis_categorias-testing1.csv')




'''Similaridad coseno tendencias'''

response_tend = tfidf.transform([activismo_laboral , adultez_emergente , anonimato, aprendizaje_conectado, aprendizaje_invertido, avances_en_el_envejecimiento, blockchain, cajas_de_suscripcion, cambio_de_privacidad, ciudades_inteligentes, coches_autonomos, control_por_voz, coworking, datos_en_todas_partes, desenchufado, desigualdad_de_ingresos, drones, economia_colaborativa, fandom, fast_casual, gamificacion, impacto_colectivo, influencia_de_la_empresa, insignias, inteligencia_artificial, internet_de_las_cosas, juguetes_conectados, lectura_corta, lugares_creativos, micromovilidad, movimiento_maker, nativos_digitales, procesamiento_de_diseno, realidad_virtual, reconocimiento_facial, renta_basica, repensar_lo_rural, resiliencia, robots, tecnologia_haptica, urbanizacion, venta_al_por_menor_experiencial])
cosine_similarity_response_tend =  cosine_similarity(response_tend, tfs)
coseno_tend = pd.DataFrame(cosine_similarity_response_tend.T, index = tesis_estudiantes_limpios["title"], columns = ["activismo_laboral ","adultez_emergente ","anonimato","aprendizaje_conectado","aprendizaje_invertido","avances_en_el_envejecimiento","blockchain","cajas_de_suscripcion","cambio_de_privacidad","ciudades_inteligentes","coches_autonomos","control_por_voz","coworking","datos_en_todas_partes","desenchufado","desigualdad_de_ingresos","drones","economia_colaborativa","fandom","fast_casual","gamificacion","impacto_colectivo","influencia_de_la_empresa","insignias","inteligencia_artificial","internet_de_las_cosas","juguetes_conectados","lectura_corta","lugares_creativos","micromovilidad","movimiento_maker","nativos_digitales","procesamiento_de_diseno","realidad_virtual","reconocimiento_facial","renta_basica","repensar_lo_rural","resiliencia","robots","tecnologia_haptica","urbanizacion","venta_al_por_menor_experiencial"
])

'''
#ind = np.arange[len(list_of_dfs)]
tesis_periodismo = coseno.sort_values(["periodismo"], ascending=False)
tesis_publicidad = coseno.sort_values(["publicidad"], ascending=False)
tesis_comunicacion_organizacional = coseno.sort_values(["comunicacion_organizacional"], ascending=False)
tesis_creacion_y_produccion_audiovisual= coseno.sort_values(["creacion_y_produccion_audiovisual"], ascending=False)
tesis_creacion_y_produccion_editorial_y_multimedial= coseno.sort_values(["creacion_y_produccion_editorial_y_multimedial"], ascending=False)
tesis_creacion_y_produccion_sonora_y_radiofonica= coseno.sort_values(["creacion_y_produccion_sonora_y_radiofonica"], ascending=False)
'''

coseno_tend.to_csv('similitud_tesis_tendencias-testing2.csv')



print("Analisis en curso. Espere un momento...")







'''
print(periodistas_pajaro.iloc[:5,0:3])
print(periodistas_zully.iloc[:5,1:2])
print(periodistas_comu.iloc[:5,2:3])


#imprimir similitud conseno
with open("output_similitud.txt","a",encoding='utf8') as f:
        print(periodistas_pajaro.iloc[:5,0:1],file=f)
        print(periodistas_pajaro.iloc[:5,0:1],file=f)
'''
