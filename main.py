from flask import Flask, render_template, request, jsonify
from werkzeug import secure_filename
import os.path
from gensim import corpora
from gensim.models import LsiModel
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from gensim.models import TfidfModel
import io
import nltk
import base64
import json
from PIL import Image 

app = Flask(__name__)

@app.route("/index")
def load_template():
	return render_template("index.html")

@app.route("/generate_stats", methods = ['GET', 'POST'])
def generate_stats():
	if request.method == 'POST':
		f = request.files['file']
		json_data = preprocess_data(f)
		return json_data

@app.route("/generate_analysis", methods = ['GET', 'POST'])
def generate_analysis():
	if request.method == 'POST':
		f = request.files['file']
		json_data = analysis(f)
		return json_data

def compute_coherence_values(corpus,dictionary, tfidf, stemmed_title_words, stop, start=2, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, stop, step):
        # generate LSA model
        lsi = LsiModel(tfidf[corpus], id2word=dictionary, num_topics=num_topics)
        model_list.append(lsi)
        
        cm = CoherenceModel(model=lsi, texts=stemmed_title_words, dictionary=dictionary, coherence='c_v')
        coherence_values.append(cm.get_coherence())
    return model_list, coherence_values

def analysis(f):
	df = pd.read_csv(f,encoding = "ISO-8859-1")
	nltk.download('stopwords')
	tokenizer = RegexpTokenizer(r'\w+')
	# create English stop words list
	en_stop = set(stopwords.words('english'))
	# Create p_stemmer of class PorterStemmer
	p_stemmer = PorterStemmer()
	# list for tokenized documents in loop
	texts = []
	title_tokens = []
	# loop through document list
	for i in df['Title']:
		# clean and tokenize document string
		#print (i)
		raw = str(i).lower()
		tokens = tokenizer.tokenize(raw)
		title_tokens.append(tokens)
		stopped_tokens = [i for i in tokens if not i in en_stop]
		stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
		texts.append(stemmed_tokens)
	stemmed_title_words = texts
	word_counts = pd.Series(np.concatenate(stemmed_title_words)).value_counts()
	singular_words = set(word_counts[pd.Series(np.concatenate(stemmed_title_words)).value_counts() == 1].index)
	stemmed_title_common_words = [[word for word in title if word not in singular_words] for title in stemmed_title_words]
	non_empty_indices = [i for i in range(len(stemmed_title_common_words)) if len(stemmed_title_common_words[i]) > 0]
	stemmed_title_common_words_nonnull = np.asarray(stemmed_title_common_words)[non_empty_indices]
	classifiable_titles = np.asarray(title_tokens)[non_empty_indices]
	dictionary = corpora.Dictionary(stemmed_title_common_words_nonnull)
	corpus = [dictionary.doc2bow(text) for text in stemmed_title_common_words_nonnull]
	tfidf = TfidfModel(corpus)
	
	
	
	model_list, coherence_values = compute_coherence_values(corpus,dictionary, tfidf, stemmed_title_words, 12, start=2, step=3)
	x = range(2, 12, 3)
	plt.plot(x, coherence_values)
	plt.xlabel("Number of Topics")
	plt.ylabel("Coherence score")
	plt.legend(("coherence_values"), loc='best')
		
	idx = coherence_values.index(max(coherence_values))
	lsi = model_list[idx]
	corpus_lsi = lsi[tfidf[corpus]]
	classifications = [np.argmax(np.asarray(corpus_lsi[i])[:,1]) for i in range(len(stemmed_title_common_words_nonnull))]
	topics = pd.DataFrame({'topic': classifications, 'title': classifiable_titles})
	topic_num = (idx*3)+2
	list_clusters = []
	for i in range(topic_num):
		cluster_titles = []
		data = topics.query('topic == @i')
		if data.empty == False:
			m = pd.Series(data['title'])
			text = " "
			for j in m.values:
				insert_text = text.join(j)
				if insert_text not in cluster_titles:
					cluster_titles.append(insert_text)
		if(len(cluster_titles) > 0):
			list_clusters.append({"cluster_id":i,"titles":cluster_titles})
	plt.savefig('C:\\Hackathon\\patent\\static\\plot.png')
	return {"clusters":list_clusters}
		
def preprocess_data(file):
	df = pd.read_csv(file,encoding = "ISO-8859-1")
	input_df = df[['Jurisdiction','Kind','Publication Number','Publication Date','Publication Year',
				'Title','Applicants','Inventors','Type','Abstract','Main Claim','Simple Family Size',
				'IPCR Classifications']]
	expand_df = explode_str(input_df,'IPCR Classifications', ';;')
	pub_no = "Publication Number"
	df = expand_df[pub_no].str.rsplit(' ',1).to_frame()
	expand_df[['Publication Number','temp']] = pd.DataFrame(df[pub_no].values.tolist(), index= df.index)
	expand_df = expand_df.drop(['temp'], axis=1)
	var = 'IPCR Classifications'
	new = expand_df[var].str.split("/", n = 1, expand = True)
	expand_df['IPCR Classifications']= new[0]
	df_apr = expand_df.drop(['Jurisdiction','Kind','Publication Date','Publication Year',
							'Title','Applicants','Inventors','Type','Abstract','Main Claim',
							'Simple Family Size'], axis =1)
	unq_codes = df_apr[var].unique()
	d_step1={}
	for code in unq_codes:
		v = str(code)
		tmp = [(df_apr[df_apr[var] ==code] 
            .groupby(['Publication Number'])
            .sum().unstack().reset_index().fillna(0) 
            .set_index('Publication Number'))]
		d_step1[v] = list(tmp[0]['level_0'].index)
	expand_df = explode_str(input_df,'Applicants', ';;')
	d={}
	a = expand_df['Jurisdiction'].value_counts()
	d['Jurisdiction'] = list(zip(a,a.index))
	a = expand_df['Kind'].value_counts()
	d['Kind'] = list(zip(a,a.index))
	a = expand_df['Inventors'].value_counts()
	d['Inventors'] = list(zip(a,a.index))
	a = expand_df['Applicants'].value_counts()
	d['Applicants'] = list(zip(a,a.index))
	a = expand_df['Type'].value_counts()
	d['Type'] = list(zip(a,a.index))
	a = expand_df['Simple Family Size'].value_counts()
	d['Simple Family Size'] = list(zip(a,a.index))
	return d

def explode_str(df, col, sep):
    s = df[col]
    i = np.arange(len(s)).repeat(s.str.count(sep) + 1)
    return df.iloc[i].assign(**{col: sep.join(s).split(sep)})

def explode_list(df, col):
    s = df[col]
    i = np.arange(len(s)).repeat(s.str.len())
    return df.iloc[i].assign(**{col: np.concatenate(s)})

if __name__ == "__main__":
  app.run()