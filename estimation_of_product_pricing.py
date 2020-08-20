import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mat

#import psycopg2

import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import datetime
import scipy.stats

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


import sys  

sys.path.insert(0, '/Users/GuntherUlvanget/country_life/price_consult')

import boot_strap_sample
import warnings
warnings.filterwarnings('ignore')

cl_top_seller = pd.read_csv("/Users/GuntherUlvanget/country_life/country_life_spins/Country Life Top Seller's Price File.csv", sep = ",")
estimation_of_product_pricing.clean_CL_UPC(cl_top_seller)

df = pd.read_csv("/Users/GuntherUlvanget/country_life/country_life_spins/Retailer_Report_24 Weeks_7_12_2020.csv", 
                 sep = ",", na_values="-", usecols= [ 'Geography', 'Item UPC', 'Brand', 'Subcategory',
       'Dollars', 'Dollars, Yago', 'Units', 'Units, Yago', 'Dollars SPP', 'Dollars SPP, Yago',  'ARP',
       'ARP, Yago',   'Base ARP','Base ARP, Yago'])
df.rename(columns =  {'Item UPC': 'Item_UPC', 
                      'Dollars, Yago':'Dollars_Yago', 
                      'Units, Yago':'Units_Yago', 
                      'Dollars SPP':'Dollars_SPP', 
                      'Dollars SPP, Yago': 'Dollars_SPP_Yago', 
                      'ARP, Yago': 'ARP_Yago',  
                      'Base ARP': 'Base_ARP',
                      'Base ARP, Yago':'BaseARP_Yago'},
         inplace= True)

estimation_of_product_pricing.clean_spins_upc(df)
items =[item for item in cl_top_seller["Item#"] ]





def clean_CL_UPC(df):
	#to match items in SPINS data, CL starts with Leading zero

	df["UPC"] = df["UPC"].apply(str)
	df["UPC"] = df["UPC"].apply(lambda x : x[:-1])
	df["UPC"] = df["UPC"].apply(lambda x : "0" + x)
	column_name = "UPC"
	first = df.pop(column_name)
	df.insert(0, column_name, first)
	df["MSRP$"] = df["MSRP$"].str.lstrip('$')
	df["WSP$"] = df["WSP$"].str.lstrip('$')
	df["MSRP$"] = df["MSRP$"].apply(float)
	df["WSP$"] = df["WSP$"].apply(float)
	df["Unit Size"] = df["Unit Size"].apply(int)
    
    
def process_data(file_path_todata):
    df = pd.read_csv(file_path_todata, 
        sep = ",", 
        na_values = "-",
       usecols= [  'Geography', 'Item_UPC', 'Brand', 'Subcategory',
           'AGE', 'FORM', 'FUNCTIONAL_INGREDIENT', 'FLAVOR', 'HEALTH_FOCUS',
        'Group', 'SIZE', 'Dollars', 'Dollars_Yago', 'Units', 'Units_Yago', 'Dollars_SPP', 'ARP',
        'ARP_Yago',  'Base_ARP',
        'Base_ARP_Yago']
         )

    df.dropna(axis = 0, how = "all", inplace=True)
    return df

def clean_spins_upc(df):
	#take in a dataframe, expand the number of columns into a temp
	#datafram
	temp = df["Item_UPC"].str.rsplit(" ", n=1, expand = True)
	#add columns  to main dataframe from Temp
	df.insert(0, "UPC", temp[1])
	df.insert(1, "Item", temp[0])

	#drop original combined column
	df.drop(columns= ["Item_UPC"], inplace=True)
	#SPINS data adds a leading zero to UP and includes hypen. 
	#neither are needed for analysis.
	df["UPC"]= df["UPC"].apply(lambda x : x[1:] if x.startswith("0") else x)
	df["UPC"]=df["UPC"].apply(lambda x :x.replace('-', ''))
	col_names = df.columns
	for cols in col_names:
	    df[cols] = df[cols].apply(lambda x: x.lower() if type(x) == str else x)
        
def sku_selcetor_ingredient(FUNCTIONAL_INGREDIENT, df):
	'''Functional ingredient can be a shortened description like cal for calcium
	or mag for magnesium. this will select all Items that contain the string entered'''
	cl_df = df[df["Brand"].str.contains("country life")]
	temp_df = cl_df[cl_df["Item"].str.contains(FUNCTIONAL_INGREDIENT)]
	
	return temp_df


def sku_singleton(item_id,df, brand = "country life"):
    '''Select a single SKU to examine, item ID can be either full UPC or catalog ID
    df = the dataframe you are subsetting'''
    df = df[df["Brand"].str.contains(brand)]
    df = df[df["UPC"].str.contains(item_id)].reset_index()
    df.dropna(inplace = True)
    #df["ARP_percent_off_MSRP"] =  df["ARP"].apply(lambda x: (1-(x/get_MSRP("item_id")))*100)
    df.loc[:, "ARP_percent_off_MSRP"] =  df.loc[:,"ARP"].apply(lambda x: (1-(x/get_MSRP(item_id)))*100)
    #df['bins'] = pd.cut(df['ARP_percent_off_MSRP'],bins=[0,5,10,15,20,25,30,35,40], labels=["0-5%","5-10%","10-15%","15-20%","20-25%","25-30%", "30-35%","35-40"])
    df.loc[:,'bins'] = pd.cut(df['ARP_percent_off_MSRP'],bins=[0,5,10,15,20,25,30,35,40], labels=["0-5% off","5-10% 0-5% off","10-15% off","15-20% off","20-25% off","25-30% off", "30-35% off","35-40% off"])
    
    return df


def get_MSRP(item_id ):
	temp = cl_top_seller[cl_top_seller.loc[:,"UPC"].str.contains(item_id)].reset_index()
	return temp["MSRP$"][0]


def sub_set_metrics_of_interest(df, metric_to_use = ['ARP',"Base_ARP"]):
	return df.loc[df["Metric"].isin(metric_to_use)].reset_index()

def easy_melt(df):
	return df.melt(id_vars=["UPC", "Item","Geography",'Brand', 'Subcategory'], var_name = "Metric" , value_name='amount')

def get_discount_proportions(df):
    ''' Takes in a dataframe, returns a smaller dataframe'''
    group_by_df =df.groupby("bins").size()
    #groups are stored in the index, values in series
    #extract groups
    groups = pd.DataFrame(group_by_df).index
    final = group_by_df.to_frame().join(groups.to_frame())
    #rename column 0 for extracting data in pie chart
    final.rename(columns = {0:"count"}, inplace = True )
    return final

def subset_for_boxplots(df):
    metric_to_use = ['ARP',"Base_ARP"]
    df = df[df["Metric"].isin(metric_to_use)].reset_index()

    return df

def cl_box_plot(df):
    fig = px.box(df, x="Metric", y="amount", points="all", title = df["Item"][0])
    fig.add_trace(go.Scatter(x=['ARP', 'Base_ARP'], y=[get_MSRP(df.loc[0,"UPC"]),get_MSRP(df.loc[0,"UPC"])], mode="lines", name="MSRP"))
    fig.show()
    
def make_plots(df,pie_frame):
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "box"}, {"type": "pie"}]],
    )

    fig.add_trace(go.Box( x=df.Metric, y=df.amount, boxpoints ="all", jitter =0.5,
                         boxmean = True, showlegend=False), row=1, col=1)
    #adds a line to reprsentMSRP
    fig.add_trace(go.Scatter(x=['ARP', 'Base_ARP'], y=[get_MSRP(df.loc[0,"UPC"]),get_MSRP(df.loc[0,"UPC"])],
                             mode="lines", name="MSRP", line=dict(color="#A9A9A9") ))
    
    #generates pie chart from calculated proportion data
    fig.add_trace(go.Pie(values=pie_frame["count"], labels =pie_frame["bins"], showlegend=True,
                         title_text ="Proportion MSRP Discounts"),
                  row=1, col=2)
    # Update xaxis properties
    fig.update_xaxes(title_text="Distribution of ARP vs Base ARP", row=1, col=1)
    #fig.update(title_text="Proportion of ARP as percentage discount",  row=1, col=2)

    fig.update_yaxes(title_text="Dollars", row=1, col=1)
    #fig.update_yaxes(title_text="Proportion of ARP as percentage discount",  row=1, col=2)

    #label the plot
    fig.update_layout(title_text=df["Item"][0], height=600)
    fig.update_layout(height=600, showlegend=True)

    fig.show()
    

def single_product_analysis(item_id):
    single_sku_df = sku_singleton(str(item_id),df)
    pie_frame = get_discount_proportions(single_sku_df)

    melted_df = easy_melt(single_sku_df)



    subset_df_forplot = subset_for_boxplots(melted_df)

    make_plots(subset_df_forplot, pie_frame)
    boot_strap_sample.plot_bootstrap_mean(single_sku_df["ARP"].to_numpy())
    boot_strap_sample.plot_bootstrap_median(single_sku_df["ARP"].to_numpy(), test_statistic=np.median)