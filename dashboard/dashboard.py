import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans

st.title("Bike-sharing Dashboard")
st.write("Analisis tren penggunaan sepeda dan pengaruh kondisi cuaca terhadap jumlah peminjaman sepeda.")

def load_data():
    file_path = os.path.join(os.getcwd(), 'dashboard', 'all_data.csv')
    data = pd.read_csv('dashboard/all_data.csv', parse_dates=['dteday'])  
    return data

def display_filtered_data(selected_date, hour_df):
    filtered_data = hour_df[hour_df['dteday'] == pd.to_datetime(selected_date)]
    st.write(f"### Data Peminjaman Sepeda untuk Tanggal {selected_date}")
    st.dataframe(filtered_data)

def display_heatmap(hour_df):
    st.write("### Rata-rata Peminjaman Sepeda Berdasarkan Hari dalam Seminggu dan Jam")
    pivot_data = hour_df.pivot_table(values='cnt_x', index='weekday_x', columns='hr', aggfunc='mean')

    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_data, cmap="YlGnBu", annot=True, fmt=".0f", cbar_kws={'label': 'Rata-rata Peminjaman Sepeda'})
    plt.title('Rata-rata Peminjaman Sepeda Berdasarkan Hari dalam Seminggu dan Jam')
    plt.xlabel('Jam dalam Sehari')
    plt.ylabel('Hari dalam Seminggu')
    plt.xticks(ticks=range(24), labels=[f'{i}:00' for i in range(24)], rotation=45)
    plt.yticks(ticks=range(7), labels=['Sen', 'Sel', 'Rab', 'Kam', 'Jum', 'Sab', 'Min'], rotation=0)
    st.pyplot(plt)

def display_daily_total(hour_df):
    st.write("### Total Peminjaman Sepeda per Hari")
    daily_data = hour_df.groupby('dteday')['cnt_x'].sum().reset_index()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=daily_data, x='dteday', y='cnt_x', marker='o', color='b')
    plt.title('Total Peminjaman Sepeda per Hari')
    plt.xlabel('Tanggal')
    plt.ylabel('Total Sepeda yang Dipinjam')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

def display_scatter_temperature(hour_df):
    st.write("### Jumlah Peminjaman Sepeda terhadap Temperatur dengan Warna Berdasarkan Kelembapan")
    plt.figure(figsize=(10, 5))
    sc = plt.scatter(hour_df['temp_x'], hour_df['cnt_x'], c=hour_df['hum_x'], cmap='coolwarm', alpha=0.6)
    plt.colorbar(sc, label="Kelembapan")
    plt.title("Jumlah Peminjaman Sepeda terhadap Temperatur dengan Warna Berdasarkan Kelembapan")
    plt.xlabel("Temperatur")
    plt.ylabel("Jumlah Peminjaman")
    st.pyplot(plt)

def display_clustering(hour_df):
    st.write("### Clustering Pengguna Sepeda Berdasarkan Jam dan Hari")
    X = hour_df[['hr', 'weekday_x']]
    kmeans = KMeans(n_clusters=3)
    hour_df['cluster'] = kmeans.fit_predict(X)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=hour_df, x='hr', y='weekday_x', hue='cluster', palette='viridis')
    plt.title('Clustering Pengguna Sepeda Berdasarkan Jam dan Hari')
    plt.xlabel('Jam')
    plt.ylabel('Hari dalam Minggu')
    st.pyplot(plt)

hour_df = load_data()

st.sidebar.header("Filter Tanggal")
selected_date = st.sidebar.date_input("Pilih tanggal", hour_df['dteday'].min())

display_filtered_data(selected_date, hour_df)

display_heatmap(hour_df)

display_daily_total(hour_df)

display_scatter_temperature(hour_df)

display_clustering(hour_df)

st.write("Dashboard sederhana untuk Analisis tren penggunaan sepeda dan pengaruh kondisi cuaca terhadap jumlah peminjaman sepeda.")
st.caption("Copyright © wahyu160 2024")