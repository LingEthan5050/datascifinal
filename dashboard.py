import pandas as pd
import numpy as np
from flask import Flask, render_template, request, send_file
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest

data_path = 'Percent_Change_in_Consumer_Spending.csv'
data_df = pd.read_csv(data_path)
data_df['Date'] = pd.to_datetime(data_df['Date'], errors='coerce')

app = Flask(__name__)

total_entries = data_df.shape[0]

sales_over_time_df = data_df.groupby('Date')['All merchant category codes spending'].mean().reset_index()
sales_over_time_fig = px.line(
    sales_over_time_df,
    x="Date",
    y="All merchant category codes spending",
    title="Average Consumer Spending Over Time",
    markers=True
)
sales_over_time_fig.update_layout(
    plot_bgcolor='rgb(0,0,0,0)',
    paper_bgcolor='rgb(0,0,0,0)',
    xaxis_title="Date",
    yaxis_title="Spending Change (%)",
    margin=dict(l=40, r=40, t=40, b=40),
    font=dict(color='#FFFFFF', size=10)
)
sales_over_time = pio.to_html(sales_over_time_fig, full_html=False, config={'displayModeBar': False})

categories = [
    'Accommodation and food service (ACF) spending',
    'Arts, entertainment, and recreation (AER)  spending',
    'General merchandise stores (GEN) and apparel and accessories (AAP) spending',
    'Grocery and food store (GRF)  spending',
    'Health care and social assistance (HCS) spending ',
    'Transportation and warehousing (TWS)  spending'
]
category_means = data_df[categories].mean()
spending_distribution_fig = go.Figure(go.Pie(
    labels=category_means.index,
    values=category_means.values,
    hole=0.3,
    marker=dict(colors=px.colors.qualitative.Plotly),
    textinfo='label+percent'
))
spending_distribution_fig.update_layout(
    showlegend=True,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),  
    margin=dict(t=0, b=0, l=0, r=0)
)

spending_distribution = pio.to_html(spending_distribution_fig, full_html=False, config={'displayModeBar': False})

data_df['Month'] = data_df['Date'].dt.month
seasonal_trends_df = data_df.groupby('Month')[categories].mean().reset_index()
seasonal_trends_fig = px.line(
    seasonal_trends_df,
    x="Month",
    y=categories,
    title="Seasonal Trends in Consumer Spending",
    markers=True
)
seasonal_trends_fig.update_layout(
    plot_bgcolor='rgb(0,0,0,0)',
    paper_bgcolor='rgb(0,0,0,0)',
    xaxis_title="Month",
    yaxis_title="Spending Change (%)",
    margin=dict(l=40, r=40, t=40, b=40),
    font=dict(color='#FFFFFF', size=10)
)
seasonal_trends = pio.to_html(seasonal_trends_fig, full_html=False, config={'displayModeBar': False})

isolation_forest = IsolationForest(contamination=0.05, random_state=42)
data_df['Anomaly'] = isolation_forest.fit_predict(data_df[categories].fillna(0))
anomalies_df = data_df[data_df['Anomaly'] == -1]
anomalies_fig = px.scatter(
    anomalies_df,
    x="Date",
    y="All merchant category codes spending",
    title="Anomalies in Consumer Spending",
    color="Anomaly",
    color_discrete_sequence=["red"]
)
anomalies_fig.update_layout(
    plot_bgcolor='rgb(0,0,0,0)',
    paper_bgcolor='rgb(0,0,0,0)',
    xaxis_title="Date",
    yaxis_title="Spending Change (%)",
    margin=dict(l=40, r=40, t=40, b=40),
    font=dict(color='#FFFFFF', size=10)
)
anomalies_chart = pio.to_html(anomalies_fig, full_html=False, config={'displayModeBar': False})


@app.route("/")
@app.route("/home")
def home():
    return render_template(
        "home.html",
      
    )



@app.route("/dashboard")
def dashboard():
    return render_template(
        "testing.html",
        total_entries=total_entries,
        sales_over_time=sales_over_time,
        spending_distribution=spending_distribution,
        seasonal_trends=seasonal_trends,
        anomalies_chart=anomalies_chart
    )
@app.route("/total_entries")   
def total_entries():
    table_html = data_df.to_html(
        classes="table table-hover table-striped table-bordered", 
        index=False, 
        border=0
    )
    return render_template(
        "total_entries.html",
        table_html=table_html
    )



@app.route("/sales_time")       
def sales_time():
    return render_template(
        "sales_time.html",
       
        sales_over_time=sales_over_time
      
    )

@app.route("/distribution")      
def distribution():
    return render_template(
        "distribution.html",
       
        spending_distribution=spending_distribution
       
    )
 
 
@app.route("/seasonal")      
def seasonal():
    return render_template(
        "seasonal.html",
        
        seasonal_trends=seasonal_trends
        
    )
    
@app.route("/anomalies")      
def anomalies():
    return render_template(
        "anomalies.html",
        
        anomalies_chart=anomalies_chart
    )
    
    
    
    
    
    
    
    
    
    

@app.route("/download")
def download():
    filtered_data_path = 'Filtered_Consumer_Spending.csv'
    data_df.to_csv(filtered_data_path, index=False)
    return send_file(filtered_data_path, as_attachment=True, download_name="Filtered_Consumer_Spending.csv")


if __name__ == '__main__':
    app.run(debug=True)
