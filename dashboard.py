import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
import numpy as np
import seaborn as sns



file_path = "Table.csv"  # Replace with your file's path

data = pd.read_csv(file_path, skiprows=3)  # Adjust skiprows based on your file structure
data.replace("(NA)", None, inplace=True)  # Replace textual "NA" with None
data.iloc[:, 4:] = data.iloc[:, 4:].apply(pd.to_numeric, errors="coerce")  # Convert columns 4+ to numeric

# Remove columns with all NaN values
data.dropna(axis=1, how="all", inplace=True)

# Remove rows with all NaN values
data.dropna(axis=0, how="all", inplace=True)

# Remove rows that are just headers
headers_to_remove = [
    "Real dollar statistics",
    "Current dollar statistics (millions of dollars)",
    "Real per capita dollar statistics (constant 2017 dollars)",
    "Per capita current dollar statistics",
    "Price indexes",
    "Employment"
]
data = data[~data["Description"].isin(headers_to_remove)]  # Exclude rows with these descriptions

#Save Cleaned Data
data.to_csv("cleaned_data.csv", index=False)

cleaned_data = pd.read_csv("cleaned_data.csv")


app = Flask(__name__)

#Homepage
@app.route("/")
def landing():
    print("Landing page accessed")
    return render_template("landing.html")





### Dataset Filtering Page

###The filtering page essentially allows us to be able to view raw data and explore it. Users can focus 
### on specific subsets of data that interest them such as specific U.S. States
###or regarding specific years. This allows for the raw data tables to be as transparent as possible by using customized queries that users can set. 



@app.route("/filter")
def index():
    print("Filter page accessed")
    state = request.args.get("state")
    year = request.args.get("year")

    filtered_data = cleaned_data
    if state:
        filtered_data = filtered_data[filtered_data["GeoName"].str.contains(state, case=False, na=False)]
    if year:
        year_column = str(year)
        if year_column in filtered_data.columns:
            filtered_data = filtered_data[["GeoFips", "GeoName", "Description", year_column]]

    subset_data = filtered_data.drop(columns=["GeoFips"], errors="ignore")

    return render_template(
        "base.html",
        columns=subset_data.columns,
        data=subset_data.values.tolist(),
        state=state,
        year=year,
    )


###Exploratory Data Analysis
###By using exploratory data analysis, we can have a better sense to understand trends and patterns in data. Examples include
###visualizing GDP growth over a state of time or to track personal income trends. In addition, it can be shown to display 
###economic growth, recession periods or outliers. 


@app.route("/eda", methods=["GET", "POST"])
def eda():
    selected_state = request.form.get("state", "United States")
    selected_metric = request.form.get("metric", "Real GDP (millions of chained 2017 dollars) 1")
    selected_year = request.form.get("year", "2021")

    filtered_data = data[
        (data["GeoName"] == selected_state) & (data["Description"].str.contains(selected_metric, case=False, na=False))
    ]

    
    img = None
    if not filtered_data.empty:
        years = [str(year) for year in range(1998, 2024) if str(year) in data.columns]
        values = filtered_data.iloc[0][years].dropna()

        plt.figure(figsize=(10, 5))
        plt.plot(values.index, values.values, marker="o")
        plt.title(f"{selected_metric} in {selected_state} Over Time")
        plt.xlabel("Year")
        plt.ylabel(selected_metric)
        plt.xticks(rotation=45)

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img = base64.b64encode(buf.getvalue()).decode()
        buf.close()
        plt.close()

    return render_template(
        "eda.html",
        states=data["GeoName"].dropna().unique(),
        metrics=data["Description"].dropna().unique(),
        years=[str(year) for year in range(1998, 2024)],
        selected_state=selected_state,
        selected_metric=selected_metric,
        selected_year=selected_year,
        plot_url=img,
    )
###Linear Regression will be used to help us predict the future values for specific states that the user chooses. 
###Essentially adds some insights into the future and can potentially assist / make some informed decisions. 
###As such it showcases the application of ML we learned in class while helping predict future trends in both a
###graph and table aspect. 


    
@app.route("/predict", methods=["GET", "POST"])
def predict():
    selected_state = request.form.get("state", "United States")
    selected_metric = request.form.get("metric", "Real GDP (millions of chained 2017 dollars) 1")
    forecast_years = 5  # Number of years to predict

    # Filter data for the selected state and metric
    filtered_data = data[
        (data["GeoName"] == selected_state) & (data["Description"].str.contains(selected_metric, case=False, na=False))
    ]

    img = None
    prediction_message = None
    prediction_table = None
    if not filtered_data.empty:
        # Extract year columns and their corresponding values
        years = [int(col) for col in data.columns if col.isdigit()]
        values = filtered_data.iloc[0][[str(year) for year in years]].dropna()

        try:
            # Prepare data for Linear Regression
            X = np.array(years[:len(values)]).reshape(-1, 1)  # Years as features
            y = values.values  # Metric values as targets
            future_years = np.array([years[-1] + i for i in range(1, forecast_years + 1)]).reshape(-1, 1)

            # Train the Linear Regression model
            model = LinearRegression()
            model.fit(X, y)

            # Predict future values
            predictions = model.predict(future_years)

            # Create a table of predicted values
            prediction_table = pd.DataFrame({
                "Year": future_years.flatten(),
                "Predicted Value": predictions
            }).round(2)

            # Plot historical data and predictions
            plt.figure(figsize=(10, 5))
            plt.plot(years[:len(values)], y, marker="o", label="Historical Data")
            plt.plot(future_years.flatten(), predictions, marker="o", linestyle="--", label="Predicted Data")
            plt.title(f"Linear Regression Prediction for {selected_metric} in {selected_state}")
            plt.xlabel("Year")
            plt.ylabel("Metric Value")
            plt.legend()
            plt.xticks(rotation=45)

            # Save plot to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            img = base64.b64encode(buf.getvalue()).decode()
            buf.close()
            plt.close()
        except Exception as e:
            prediction_message = f"Error in prediction: {e}"

    return render_template(
        "predict.html",
        states=data["GeoName"].dropna().unique(),
        metrics=data["Description"].dropna().unique(),
        selected_state=selected_state,
        selected_metric=selected_metric,
        plot_url=img,
        table=prediction_table.to_dict("records") if prediction_table is not None else None,
        message=prediction_message,
    )



###Heat Map generated to show correlations between economic metrics for certain states or all states. 
###Uncovers relationships between variables and shows metrics that are highly related and can influence one
###another. Will later help with building better predictive models or understanding economic dynamics


@app.route("/heatmap", methods=["GET", "POST"])
def heatmap():
    selected_state = request.form.get("state", None)  # Default to all states

    # Filter data for the selected state, if provided
    filtered_data = data.copy()
    if selected_state:
        filtered_data = data[data["GeoName"] == selected_state]

    # Calculate correlations for year columns only
    years = [col for col in filtered_data.columns if col.isdigit()]
    corr = filtered_data[years].corr()

    # Generate the heatmap plot
    img = None
    if not corr.empty:
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title(f"Correlation Heatmap ({selected_state or 'All States'})")
        plt.tight_layout()

        # Save plot to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img = base64.b64encode(buf.getvalue()).decode()
        buf.close()
        plt.close()

    return render_template(
        "heatmap.html",
        states=data["GeoName"].dropna().unique(),
        selected_state=selected_state,
        plot_url=img,
    )



###Compares metrics between two states for specific year. Allow users to evaluable differences between regions or disparities
###Helps identify which policies / conditions may contribute to better economic outcomes in one state over another.


@app.route("/compare", methods=["GET", "POST"])
def compare():
    state1 = request.form.get("state1", None)
    state2 = request.form.get("state2", None)
    year = request.form.get("year", None)

    img = None
    message = None

    if state1 and state2 and year:
        # Filter data for the two states
        data1 = data[(data["GeoName"] == state1) & (data[year].notna())]
        data2 = data[(data["GeoName"] == state2) & (data[year].notna())]

        if not data1.empty and not data2.empty:
            plt.figure(figsize=(10, 6))
            plt.bar(data1["Description"], data1[year], label=state1, alpha=0.7)
            plt.bar(data2["Description"], data2[year], label=state2, alpha=0.7)
            plt.title(f"Comparison of {state1} and {state2} for {year}")
            plt.ylabel("Metric Value")
            plt.xticks(rotation=45, ha="right")
            plt.legend()

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            img = base64.b64encode(buf.getvalue()).decode()
            buf.close()
            plt.close()
        else:
            message = "Data not available for the selected states or year."

    return render_template(
        "compare.html",
        states=data["GeoName"].dropna().unique(),
        years=[col for col in data.columns if col.isdigit()],
        state1=state1,
        state2=state2,
        year=year,
        plot_url=img,
        message=message,
    )




if __name__ == "__main__":
    app.run(debug=True, port=5000)
