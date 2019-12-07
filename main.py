from flask import Flask, render_template, request
from werkzeug import secure_filename
import pandas as pd
import numpy as np
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
	pass

		
		
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