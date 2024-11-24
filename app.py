import sys
import datetime
import warnings
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn import metrics
import plotly.express as px
from matplotlib import colors
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering
pd.set_option('display.max_columns', None)
if not sys.warnoptions:
    warnings.simplefilter("ignore")


# Load default sample dataset
def load_sample_data():
    # Replace 'sample.csv' with the path to your default CSV file
    sample_data = pd.read_csv("marketing_campaign.csv", sep="\t")
    return sample_data

# App title
st.title("Customer Segmentation Application")
st.write("""
This application allows companies or individuals to upload customer data in CSV format, analyze their customer base, and predict customer segments.
""")

# File uploader
uploaded_file = st.file_uploader("Upload your customer data (CSV file)", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded file into a Pandas DataFrame
    try:
        data = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
    except Exception as e:
        st.error(f"An error occurred while reading the file: {e}")
else:
    # Load sample data if no file is uploaded
    st.info("No file uploaded. Using the default sample dataset.")
    data = load_sample_data()

# Display dataset information
if data is not None:
    st.write(f"**Number of rows and columns:** {data.shape}")
    st.subheader("Preview of the dataset")
    st.write(data.head())


def clean_and_engineer_data(data):
    data = data.dropna()
    # Convert Dt_Customer to datetime and calculate Customer_For
    data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"], format="%d-%m-%Y", errors="coerce")

    max_date = data["Dt_Customer"].max()
    data["Customer_For"] = (max_date - data["Dt_Customer"]).dt.days


    # Calculate Age
    data["Age"] = 2021 - data["Year_Birth"]

    # Total spendings on various items
    spending_columns = [
        "MntWines", "MntFruits", "MntMeatProducts",
        "MntFishProducts", "MntSweetProducts", "MntGoldProds"
    ]
    data["Spent"] = data[spending_columns].sum(axis=1)

    # Derive living situation
    data["Living_With"] = data["Marital_Status"].replace({
        "Married": "Partner", "Together": "Partner",
        "Absurd": "Alone", "Widow": "Alone",
        "YOLO": "Alone", "Divorced": "Alone",
        "Single": "Alone"
    })

    # Calculate Children and Family Size
    data["Children"] = data["Kidhome"] + data["Teenhome"]
    data["Family_Size"] = data["Living_With"].replace({"Alone": 1, "Partner": 2}) + data["Children"]

    # Determine parenthood
    data["Is_Parent"] = np.where(data["Children"] > 0, 1, 0)

    # Segment education levels
    data["Education"] = data["Education"].replace({
        "Basic": "Undergraduate", "2n Cycle": "Undergraduate",
        "Graduation": "Graduate", "Master": "Postgraduate",
        "PhD": "Postgraduate"
    })

    # Rename spending columns for clarity
    data = data.rename(columns={
        "MntWines": "Wines", "MntFruits": "Fruits",
        "MntMeatProducts": "Meat", "MntFishProducts": "Fish",
        "MntSweetProducts": "Sweets", "MntGoldProds": "Gold"
    })

    # Create Age Group feature
    data["Age_Group"] = [
        "Young" if age <= 35 else
        "Mid-aged" if 36 <= age <= 55 else
        "Old" for age in data["Age"]
    ]

    # Drop redundant features
    to_drop = ["Marital_Status", "Dt_Customer", "Z_CostContact", "Z_Revenue", "Year_Birth", "ID"]
    data = data.drop(columns=to_drop)

    # Remove outliers
    data = data[data["Age"] < 90]
    data = data[data["Income"] < 600000]

    return data

data = clean_and_engineer_data(data)
st.subheader("Data Cleaning")
st.write(data.head())

# Streamlit başlığı
st.title("Customer Segmentation: Feature Analysis")

# Kullanıcıya hue seçimi için widget
hue_options = ["Is_Parent", "Living_With"]  # Kullanılabilecek hue seçenekleri
selected_hue = st.selectbox("Select hue for pairplot:", hue_options, index=0)

# Görselleştirme için seaborn ayarları
sns.set(rc={"axes.facecolor": "#FFF9ED", "figure.facecolor": "#FFF9ED"})
palette = ["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"]

# İncelenecek değişkenler
To_Plot = ["Income", "Recency", "Customer_For", "Age", "Spent", selected_hue]

# Pairplot oluştur
st.write(f"### Pairplot with hue: {selected_hue}")
fig = sns.pairplot(data[To_Plot], hue=selected_hue, palette=palette)

# Plot'u Streamlit'te göster
st.pyplot(fig)

# Streamlit başlığı
st.title("Customer Segmentation: Correlation Matrix")

# Korelasyon matrisini oluştur
numeric_data = data.select_dtypes(include=['number'])
corrmat = numeric_data.corr()

# Sadece üst üçgeni almak (Simetrik kısmı kaldırma)
mask = np.triu(np.ones_like(corrmat, dtype=bool))

# Seaborn ile ısı haritası görselleştirme (statik grafik)
plt.figure(figsize=(24, 15))  # Grafik boyutunu artırdık
sns.heatmap(corrmat, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, cbar_kws={'shrink': 0.8},
            annot_kws={'size': 12, 'ha': 'center', 'va': 'center'})  # Yazı boyutunu artırdık ve konumlandırmayı yaptık

# Başlık ekleyelim
plt.title("Filtered Correlation Heatmap", fontsize=16)

# Streamlit ile grafiği göster
st.pyplot(plt)
plt.clf()
def data_preprocessing(data):
    s = (data.dtypes == 'object')
    object_cols = list(s[s].index)
    LE = LabelEncoder()
    for i in object_cols:
        data[i] = data[[i]].apply(LE.fit_transform)

    ds = data.copy()
    # creating a subset of dataframe by dropping the features on deals accepted and promotions
    cols_del = ['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'Response']
    ds = ds.drop(cols_del, axis=1)
    # Scaling
    scaler = StandardScaler()
    scaler.fit(ds)
    scaled_ds = pd.DataFrame(scaler.transform(ds), columns=ds.columns)
    return scaled_ds

scaled_ds = data_preprocessing(data)

def dimensionality_reduction(scaled_ds):
    pca = PCA(n_components=3)
    pca.fit(scaled_ds)
    PCA_ds = pd.DataFrame(pca.transform(scaled_ds), columns=(["col1", "col2", "col3"]))
    return PCA_ds, scaled_ds

PCA_ds, scaled_ds = dimensionality_reduction(scaled_ds)

# Streamlit başlığı
st.title("Clustering: Elbow Method")

st.write("The elbow plot is a method used to determine the optimal number of clusters for K-means clustering by visualizing the point where the within-cluster variance starts to diminish.")

# Kullanıcıdan 'k' değerini alma
k_value = st.number_input('Enter the number of clusters (k)', min_value=1, max_value=20, value=10)

# Elbow Method (KMeans ve KElbowVisualizer kullanarak)
st.write("Elbow Method to determine the number of clusters:")
Elbow_M = KElbowVisualizer(KMeans(random_state=42), k=k_value, locate_elbow=True, timings=True)
Elbow_M.fit(PCA_ds)
# This clears the current figure
fig = Elbow_M.fig  # Burada figür nesnesini alıyoruz

# Streamlit içerisinde Elbow grafiğini görüntüle
st.pyplot(fig)

def agglomerative(PCA_ds):
    # Initiating the Agglomerative Clustering model
    AC = AgglomerativeClustering(n_clusters=4)
    # fit model and predict clusters
    yhat_AC = AC.fit_predict(PCA_ds)
    PCA_ds["Clusters"] = yhat_AC
    # Adding the Clusters feature to the orignal dataframe.
    data["Clusters"] = yhat_AC
    return data

data = agglomerative(PCA_ds)
st.write(data.head())

st.title("Distribution Of The Clusters")
# Assuming 'data' DataFrame already has a 'Clusters' column
pal = ["#682F2F", "#B9C0C9", "#9F8A78", "#F3AB60"]
plt.figure(figsize=(10, 6))  # Adjust the figure size for better display
pl = sns.countplot(x=data["Clusters"], palette=pal)
pl.set_title("Distribution Of The Clusters")

# Display the plot in Streamlit
st.pyplot(plt)

st.title("Cluster's Profile Based On Income And Spending")
# Assuming 'data' DataFrame has 'Spent', 'Income', and 'Clusters' columns
pal = ["#682F2F", "#B9C0C9", "#9F8A78", "#F3AB60"]
plt.figure(figsize=(10, 6))  # Adjust the figure size for better display

# Scatter plot showing relationship between Spending and Income, colored by Clusters
pl = sns.scatterplot(data=data, x="Spent", y="Income", hue="Clusters", palette=pal)
pl.set_title("Cluster's Profile Based On Income And Spending")

# Display the plot in Streamlit
plt.legend()
st.pyplot(plt)

st.title("Spent Distribution by Clusters")
# Assuming 'data' DataFrame has 'Spent' and 'Clusters' columns
pal = ["#682F2F", "#B9C0C9", "#9F8A78", "#F3AB60"]
plt.figure(figsize=(10, 6))  # Adjust the figure size for better display

# Swarm plot for 'Spent' based on 'Clusters'
sns.swarmplot(x=data["Clusters"], y=data["Spent"], color="#CBEDDD", alpha=0.5)

# Boxen plot for 'Spent' based on 'Clusters'
sns.boxenplot(x=data["Clusters"], y=data["Spent"], palette=pal)

# Display the plot in Streamlit
plt.title("Spent Distribution by Clusters")
st.pyplot(plt)

st.title("Count Of Promotion Accepted")
# Assuming 'data' DataFrame contains 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', and 'Clusters'
data["Total_Promos"] = data["AcceptedCmp1"] + data["AcceptedCmp2"] + data["AcceptedCmp3"] + data["AcceptedCmp4"] + data["AcceptedCmp5"]

# Define the color palette
pal = ["#682F2F", "#B9C0C9", "#9F8A78", "#F3AB60"]

# Create a count plot for the number of accepted promotions, grouped by clusters
plt.figure(figsize=(10, 6))  # Adjust the figure size for better display
pl = sns.countplot(x=data["Total_Promos"], hue=data["Clusters"], palette=pal)
pl.set_title("Count Of Promotion Accepted")
pl.set_xlabel("Number Of Total Accepted Promotions")

# Display the plot in Streamlit
st.pyplot(plt)

st.title("Number of Deals Purchased")
# Assuming 'data' DataFrame contains 'NumDealsPurchases' and 'Clusters'
# Define the color palette
pal = ["#682F2F", "#B9C0C9", "#9F8A78", "#F3AB60"]

# Create a boxen plot for the number of deals purchased, grouped by clusters
plt.figure(figsize=(10, 6))  # Adjust the figure size for better display
pl = sns.boxenplot(y=data["NumDealsPurchases"], x=data["Clusters"], palette=pal)
pl.set_title("Number of Deals Purchased")

# Display the plot in Streamlit
st.pyplot(plt)


st.title("Cluster Profiling")
# Assuming 'data' DataFrame contains the required columns and 'Clusters' is already defined
pal = ["#682F2F", "#B9C0C9", "#9F8A78", "#F3AB60"]

# List of personal features to plot
Personal = ["Kidhome", "Teenhome", "Customer_For", "Age", "Children", "Family_Size", "Is_Parent", "Education",
            "Living_With", "Age_Group"]


# Define a function to automatically generate insights based on clustering patterns
def generate_insights(data, feature, cluster_column="Clusters"):
    cluster_insights = {}

    # Iterate through each cluster and analyze its distribution
    for cluster_id in data[cluster_column].unique():
        cluster_data = data[data[cluster_column] == cluster_id]

        # Example analysis: Check the mean of the feature for each cluster
        mean_value = cluster_data[feature].mean()
        std_dev = cluster_data[feature].std()

        # Example analysis: Compare feature ranges for different clusters
        min_value = cluster_data[feature].min()
        max_value = cluster_data[feature].max()

        # Automatically generate insights based on the feature's distribution in each cluster
        cluster_insights[cluster_id] = {
            "mean_value": mean_value,
            "std_dev": std_dev,
            "min_value": min_value,
            "max_value": max_value,
            "interpretation": []
        }

        # Generate automatic insights based on feature distributions
        if mean_value > cluster_data[feature].median():
            cluster_insights[cluster_id]["interpretation"].append(
                f"Cluster {cluster_id} has a higher average {feature} than the median.")
        else:
            cluster_insights[cluster_id]["interpretation"].append(
                f"Cluster {cluster_id} has a lower average {feature} than the median.")

        if std_dev > mean_value:
            cluster_insights[cluster_id]["interpretation"].append(
                f"Cluster {cluster_id} shows high variability in {feature}.")
        else:
            cluster_insights[cluster_id]["interpretation"].append(
                f"Cluster {cluster_id} shows low variability in {feature}.")

        cluster_insights[cluster_id]["interpretation"].append(
            f"The range of {feature} in cluster {cluster_id} is from {min_value} to {max_value}.")

    return cluster_insights


# Create joint plots for each feature and generate insights
for i in Personal:
    plt.figure(figsize=(10, 6))  # Adjust figure size for better readability
    g = sns.jointplot(x=data[i], y=data["Spent"], hue=data["Clusters"], kind="kde", palette=pal)
    g.fig.suptitle(f"Jointplot of {i} and Spent", fontsize=16)  # Title for each plot
    g.fig.tight_layout()  # Adjust layout for the title to fit
    g.fig.subplots_adjust(top=0.95)  # Space for the suptitle

    # Display the plot in Streamlit
    st.pyplot(g.fig)

    # Generate insights for each cluster group based on the current feature
    insights = generate_insights(data, i)

    # Display the insights for each cluster
    for cluster_id, info in insights.items():
        st.write(f"### Cluster {cluster_id} Insights for {i}:")
        for interpretation in info["interpretation"]:
            st.write(f"- {interpretation}")
        st.write(f"Mean: {info['mean_value']:.2f}, Std Dev: {info['std_dev']:.2f}")
        st.write(f"Min: {info['min_value']}, Max: {info['max_value']}")

# Contact Information Section
st.markdown("## For Further Questions and Contact")
st.write("If you have any questions or feedback about this project, please don’t hesitate to reach out! You can contact me through the following platforms:")
st.write("📧 **Email**: [furkansukan10@gmail.com](mailto:furkansukan10@gmail.com)")
st.write("🪪 **LinkedIn**: [Furkan Sukan](https://www.linkedin.com/in/furkansukan/)")
st.write("🔗 **Kaggle**: [Furkan Sukan](https://www.kaggle.com/furkansukan)")
st.write("🐙 **GitHub**: [Furkan Sukan](https://github.com/furkansukan)")
st.write("I’d be delighted to hear your thoughts and suggestions!")