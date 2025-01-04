import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import GRU, Dense
from keras.layers import Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
import scipy.optimize as sco
import matplotlib.pyplot as plt
import random

# Fungsi untuk menghitung pengembalian yang diharapkan
def geometric_mean(series):
    return (np.prod(1 + series / 100) ** (1 / len(series)) - 1) * 100

# Fungsi untuk menghitung risk-free rate dari data BI rate
def calculate_risk_free_rate(bi_rate_data, start_date, end_date):
    # Filter BI rate data untuk mencocokkan periode data saham
    filtered_bi_rate_data = bi_rate_data[
        (bi_rate_data['Tanggal'] >= start_date) & (bi_rate_data['Tanggal'] <= end_date)
    ]
    
    # Validasi data yang difilter
    if filtered_bi_rate_data.empty:
        raise ValueError("Filtered BI rate data is empty. Ensure the date range matches the stock data period.")
    
    # Hitung geometric mean dari risk-free rate
    geometric_mean_rate = (
        np.prod(filtered_bi_rate_data['BI_7Day_RR'] ** (1 / len(filtered_bi_rate_data))) * 100
    )
    monthly_risk_free_rate = geometric_mean_rate / 12
    return monthly_risk_free_rate

# Fungsi untuk mengoptimalkan portofolio
def portfolio_optimization(returns, filtered_stocks, filtered_expected_returns, monthly_risk_free_rate):
    filtered_returns = returns[filtered_stocks]
    
    # Calculate risk (standard deviation), covariance matrix, and correlation matrix
    risks = filtered_returns.std()
    cov_matrix = filtered_returns.cov()
    corr_matrix = filtered_returns.corr()

    # Portfolio return function
    def portfolio_return(weights, filtered_expected_returns):
        return np.dot(weights, filtered_expected_returns)

    # Portfolio risk function
    def portfolio_risk(weights, cov_matrix):
        weights = np.array(weights)
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        return np.sqrt(portfolio_variance)

    # Portfolio optimization function
    def optimize_portfolio(target_return, filtered_expected_returns, cov_matrix):
        num_assets = len(filtered_expected_returns)
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'ineq', 'fun': lambda x: portfolio_return(x, filtered_expected_returns) - target_return}
        ]
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_guess = num_assets * [1. / num_assets]
        result = sco.minimize(
            portfolio_risk,
            initial_guess,
            args=(cov_matrix,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        if result.success:
            return np.round(result.x, 4)
        else:
            return None

    target_returns = np.arange(0.0, 6.0, 0.1)
    optimization_results = []

    for target_return in target_returns:
        optimal_weights = optimize_portfolio(target_return, filtered_expected_returns, cov_matrix)
        if optimal_weights is not None:
            normalized_weights = optimal_weights / np.sum(optimal_weights)
            optimal_portfolio_risk = portfolio_risk(normalized_weights, cov_matrix)
            sharpe_ratio = (target_return - monthly_risk_free_rate) / optimal_portfolio_risk
            result = {
                'Target Return (%)': target_return,
                'Portfolio Risk (Std Dev, %)': optimal_portfolio_risk,
                'Sharpe Ratio': sharpe_ratio
            }
            for j, stock in enumerate(filtered_stocks):
                result[f'Weight ({stock}) (%)'] = normalized_weights[j] * 100
            optimization_results.append(result)
    
    optimization_results_df = pd.DataFrame(optimization_results)
    optimization_results_df.set_index('Target Return (%)', inplace=True)

    max_sharpe_index = optimization_results_df['Sharpe Ratio'].idxmax()
    max_sharpe_weights = optimization_results_df.loc[max_sharpe_index]

    # Membuat DataFrame untuk bobot optimal dengan Sharpe Ratio tertinggi
    max_sharpe_weights_df = pd.DataFrame({
        'Emiten': filtered_stocks,
        'Optimal Weight (%)': max_sharpe_weights[[f'Weight ({stock}) (%)' for stock in filtered_stocks]].values
    })

    # Filter hanya bobot yang lebih besar dari 0
    max_sharpe_weights_df = max_sharpe_weights_df[max_sharpe_weights_df['Optimal Weight (%)'] > 0]
    
    return optimization_results_df, max_sharpe_weights, cov_matrix, corr_matrix, max_sharpe_weights_df

# Fungsi untuk menampilkan grafik pengembalian tahunan
def plot_annual_returns(filtered_returns, max_sharpe_weights_df):
    # Ensure the weights are aligned with the columns of filtered_returns
    weights = np.zeros(len(filtered_returns.columns))
    for i, stock in enumerate(filtered_returns.columns):
        if stock in max_sharpe_weights_df['Emiten'].values:
            weights[i] = max_sharpe_weights_df[max_sharpe_weights_df['Emiten'] == stock]['Optimal Weight (%)'].values[0] / 100

    portfolio_monthly_returns = (filtered_returns * weights).sum(axis=1)

    # Convert index to datetime and group by year
    portfolio_monthly_returns.index = pd.to_datetime(portfolio_monthly_returns.index)
    portfolio_monthly_returns = portfolio_monthly_returns.to_frame(name='Portfolio Return')
    portfolio_monthly_returns['Year'] = portfolio_monthly_returns.index.year

    # Calculate annual returns
    annual_portfolio_returns = portfolio_monthly_returns.groupby('Year')['Portfolio Return'].sum()

    # Plot the annual returns as a line chart
    plt.figure(figsize=(10, 6))
    plt.plot(annual_portfolio_returns.index, annual_portfolio_returns.values, marker='o', color='skyblue', linewidth=2, label='Portfolio Return')

    # Tambahkan nilai pada tiap titik
    for x, y in zip(annual_portfolio_returns.index, annual_portfolio_returns.values):
        plt.text(x, y + 0.5, f"{y:.2f}%", ha='center', va='bottom', fontsize=10, color='black')

    # Formatting the plot
    plt.title("Annual Portfolio Returns")
    plt.xlabel("Year")
    plt.ylabel("Total Portfolio Return (%)")
    plt.xticks(annual_portfolio_returns.index)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

    # Use Streamlit to display the plot
    st.pyplot(plt)

def download_stock_data(optimal_tickers, prices):
    """
    Mengunduh data saham harian berdasarkan daftar ticker dan periode yang ditentukan.
    
    Args:
    - optimal_tickers (list): Daftar ticker yang akan diunduh.
    - prices (pd.DataFrame): Data harga saham untuk menentukan periode.
    
    Returns:
    - stock_data_df (pd.DataFrame): Data saham yang telah digabungkan.
    """
    # Ambil periode start dan end dari data awal
    start_date = prices.index.min()  # Tanggal awal dari data bulanan
    end_date = prices.index.max() + pd.offsets.MonthEnd(0)  # Akhir bulan terakhir dari data bulanan
    
    # Cetak periode yang digunakan
    print(f"Start date: {start_date}")
    print(f"End date: {end_date}")
    
    # Mengunduh data saham untuk semua ticker dalam optimal_tickers
    stock_data = {}
    for ticker in optimal_tickers:
        try:
            # Mengunduh data saham harian untuk ticker tertentu
            df = yf.download(ticker, start=start_date, end=end_date, interval='1d')['Close']
            stock_data[ticker] = df
            print(f"Data untuk {ticker} berhasil diunduh.")
        except Exception as e:
            print(f"Error mengunduh data untuk {ticker}: {e}")
    
    # Gabungkan data ke dalam DataFrame
    stock_data_df = pd.concat(stock_data.values(), axis=1, keys=stock_data.keys())
    stock_data_df.columns = optimal_tickers  # Pastikan kolom sesuai dengan nama ticker
    
    # Tangani data yang hilang
    stock_data_df = stock_data_df.fillna(method='ffill').fillna(method='bfill')
    
    # Tampilkan DataFrame gabungan
    print(stock_data_df.head())
    
    return stock_data_df

DEFAULT = 'PICK A VALUE'

def selectbox_with_default(text, values, default=DEFAULT, sidebar=False):
    func = st.sidebar.selectbox if sidebar else st.selectbox
    return func(text, [default] + list(values))

# Fungsi untuk Membangun Model
def build_model(hidden_layers, neurons, learning_rate, time_step, dropout_rate=0.2):
    model = Sequential()
    model.add(GRU(units=neurons, return_sequences=(hidden_layers > 1), input_shape=(time_step, 1)))
    model.add(Dropout(dropout_rate)) 
    for _ in range(hidden_layers - 1):
        model.add(GRU(units=neurons, return_sequences=False))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1))  # Output layer untuk regresi
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model

# Fungsi untuk mempersiapkan data
def create_dataset(dataset, time_step=6):
    x, y = [], []
    for i in range(time_step, len(dataset)):
        x.append(dataset[i-time_step:i, 0])
        y.append(dataset[i, 0])
    return np.array(x), np.array(y)

# Fungsi utama untuk model
def run_model(data, selected_data, best_model_params):
    # Set seed untuk memastikan hasil yang konsisten
    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

    # Pastikan selected_data adalah DataFrame
    selected_data = data[selected_ticker].to_frame()

    # Normalisasi data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(selected_data)

    # Pembagian data menjadi pelatihan dan pengujian
    train_size = int(len(scaled_data) * 0.6)
    val_size = int(len(scaled_data) * 0.2)

    train_data = scaled_data[:train_size]
    val_data = scaled_data[train_size:train_size + val_size]
    test_data = scaled_data[train_size + val_size:]  # Sisanya untuk test
    
    # Mempersiapkan data dengan window size (timesteps)
    time_step = 6
    x_train, y_train = create_dataset(train_data, time_step)
    x_val, y_val = create_dataset(val_data, time_step)
    x_test, y_test = create_dataset(test_data, time_step)

    # Reshape data agar sesuai dengan input GRU
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_val = x_val.reshape(x_val.shape [0], x_val.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    # Ambil parameter terbaik
    batch_size = int(best_model_params['Batch Size'])
    neurons = int(best_model_params['Neurons'])
    hidden_layers = int(best_model_params['Hidden Layers'])
    epochs = int(best_model_params['Epochs'])
    learning_rate = float(best_model_params['Learning Rate'])
    dropout_rate = float(best_model_params['Dropout'])
    
    # Bangun model dengan parameter terbaik
    best_model = build_model(hidden_layers, neurons, learning_rate, time_step, dropout_rate)

    # Latih model dengan data pelatihan
    best_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val), shuffle=False, verbose=0)

    # Prediksi pada data pengujian
    y_test_pred = best_model.predict(x_test).flatten()

    # Denormalisasi data aktual dan prediksi
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_test_pred_denormalized = scaler.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()

    # Hitung MAPE untuk data pengujian setelah denormalisasi
    test_mape_denormalized = mean_absolute_percentage_error(y_test_actual, y_test_pred_denormalized)

    # Plot hasil prediksi vs data sebenarnya setelah denormalisasi
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(y_test_actual, label='Actual', color='blue')
    ax.plot(y_test_pred_denormalized, label='Predicted', color='red')
    ax.set_title('Actual vs Predicted - Testing Part (Denormalized)')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('CPO Production')
    ax.legend()

    # Tampilkan plot di Streamlit
    st.pyplot(fig)  # Menampilkan plot

    return test_mape_denormalized, best_model, scaler, scaled_data

def forecast_next_periods(best_model, scaler, scaled_data, time_step, forecast_periods=60):
    """
    Melakukan forecasting berdasarkan model yang sudah dilatih.

    Parameters:
        - best_model: Model GRU/LSTM yang sudah dilatih.
        - scaler: Scaler yang digunakan untuk normalisasi.
        - scaled_data: Data yang sudah dinormalisasi.
        - time_step: Window size.
        - forecast_periods: Jumlah periode yang akan diprediksi.

    Returns:
        - forecast_results_denormalized: Hasil forecast yang sudah di-denormalisasi.
    """
    last_sequence = scaled_data[-time_step:].flatten()
    forecast_results = []

    for _ in range(forecast_periods):
        next_value = best_model.predict(last_sequence.reshape(1, time_step, 1)).flatten()[0]
        forecast_results.append(next_value)
        last_sequence = np.append(last_sequence[1:], next_value)

    forecast_results_denormalized = scaler.inverse_transform(np.array(forecast_results).reshape(-1, 1)).flatten()
    return forecast_results_denormalized


def display_forecast_plot(scaled_data, forecast_results_denormalized, forecast_periods, scaler):
    """
    Menampilkan plot hasil forecast.

    Parameters:
        - scaled_data: Data input yang sudah dinormalisasi.
        - forecast_results_denormalized: Hasil forecast (denormalisasi).
        - forecast_periods: Jumlah periode forecasting.
        - scaler: Scaler untuk denormalisasi data.
    """
    y_actual_denormalized = scaler.inverse_transform(scaled_data).flatten()

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(y_actual_denormalized)), y_actual_denormalized, label='Actual', color='blue')
    plt.plot(range(len(y_actual_denormalized), len(y_actual_denormalized) + forecast_periods),
             forecast_results_denormalized, label='Forecasted', color='red')
    plt.title('Actual vs Forecasted Data')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    st.pyplot(plt)


# Misalkan optimal_tickers dan prices sudah tersedia, kamu bisa panggil fungsi ini
# stock_data_df = download_stock_data(optimal_tickers, prices)

# Sidebar Navigation
st.sidebar.title("MENU")

menu = st.sidebar.selectbox(
    "Pilih Halaman:",
    ["🏠 Home", "📚 Tutorial", "📊 Analyze"]
)

# **Home Page**
if menu == "🏠 Home":
    st.title("Selamat Datang di Aplikasi Analisis Saham 🎉")
    st.markdown(
        """
        Selamat datang di Dashboard Analisis Saham. Dashboard ini dirancang untuk membantu pengguna dalam melakukan optimasi portofolio saham menggunakan metode Markowitz, pemilihan portofolio optimal berdasarkan Sharpe Ratio, serta peramalan harga saham dengan model GRU. Pada halaman ini, pengguna akan diperkenalkan dengan fungsi utama dashboard dan panduan singkat untuk memulai analisis.

        Pengguna dapat mengeksplorasi berbagai fitur yang tersedia, termasuk pengambilan data saham secara langsung melalui input form yang terintegrasi dengan Yahoo Finance, pengaturan periode waktu data, serta visualisasi hasil analisis yang mudah dipahami. Selain itu, pengguna juga dapat menggunakan data pilihan mereka sendiri, menjadikan dashboard ini fleksibel dan dapat disesuaikan dengan kebutuhan masing-masing.

        Untuk memulai, silakan pilih menu Analyze untuk memasukkan data saham, klik tombol "Analyze" untuk mendapatkan portofolio optimal, setelah hasil optimasi keluar beralih ke fitut Forecasting dan klik tombol "Tampilkan Data" untuk melihat peramalan harga saham. Dengan tampilan yang interaktif dan informatif, dashboard ini diharapkan dapat menjadi alat bantu yang efektif dalam pengambilan keputusan investasi yang lebih akurat dan berbasis data. 
                
        👉 Navigasikan melalui **menu di sebelah kiri** untuk memulai.
        """
    )

# **Tutorial Page**
elif menu == "📚 Tutorial":
    st.title("📖 Panduan Penggunaan")
    st.markdown(
        """
        **Langkah-langkah:**
        1. Masukkan **kode emiten** yang ingin dianalisis pada halaman Analyze.
        2. Tentukan **periode waktu** yang diinginkan.
        3. Klik tombol **"Analyze"** untuk mendapatkan hasil portofolio optimal.
        4. Lihat **ringkasan data** dan analisis lebih lanjut.
        5. Pilih **kode emiten** untuk mendapatkan hasil peramalan harga saham.

        untuk penjelasan lengkap cara penggunaan dan output analisis silahkan tonton video berikut!
        """
    )
    st.image("https://via.placeholder.com/800x400.png?text=Panduan+Visual", use_container_width=True)

# **Analyze Page**
elif menu == "📊 Analyze":
    st.title("🔍 Analisis Saham")

    # Form pertama untuk input tickers dan periode waktu
    with st.form("input_form_1"):
        st.subheader("Masukkan Parameter Analisis")
        st.markdown(
            """
            💡 **Tips**:
            - Gunakan format kode emiten seperti `BBCA.JK` untuk analisis saham.
            - Periksa periode data sebelum mengunduh.
            """
        )

        # Input tickers secara manual
        selected_tickers = st.text_area(
            "Masukkan kode emiten (pisahkan dengan koma):",
            value="ACES.JK, ANTM.JK, ARTO.JK, BBCA.JK, BBNI.JK, BBRI.JK, BBTN.JK, BMRI.JK, BOGA.JK, BRIS.JK, BRMS.JK, BRPT.JK, BTPS.JK, CASA.JK, CMNT.JK, CTRA.JK, ERAA.JK, ESSA.JK, FILM.JK, INCO.JK, INKP.JK, INTP.JK, MAPI.JK, MDKA.JK, MNCN.JK, PNLF.JK, SCMA.JK, SMGR.JK, SRTG.JK, TPIA.JK",
            help="Pisahkan kode emiten dengan koma. Contoh: BBCA.JK, BBRI.JK"
        )
        selected_tickers = [ticker.strip() for ticker in selected_tickers.split(",") if ticker.strip()]

        ## Input tanggal menggunakan kalender
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=pd.Timestamp("2021-10-01"),  # Default value
                min_value=pd.Timestamp("2000-01-01"),
                max_value=pd.Timestamp("2024-12-31"),
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=pd.Timestamp("2024-10-31"),  # Default value
                min_value=pd.Timestamp("2000-01-01"),
                max_value=pd.Timestamp("2024-12-31"),
            )

        # Tombol Submit untuk form pertama
        submit_button_1 = st.form_submit_button(label="Analyze")

        if submit_button_1:
            # Mengunduh data harga saham
            prices, error = None, None
            try:
                prices = yf.download(selected_tickers, start=start_date, end=end_date, interval="1mo")['Close']
                prices = prices.dropna(how="all")  # Hapus baris dengan semua nilai NaN
            except Exception as e:
                error = str(e)

            if error:
                st.error(f"Error: {error}")
            else:
                st.success("Data harga saham berhasil diunduh.")
                st.dataframe(prices)

                # Menyimpan data harga saham dan tickers ke session_state
                st.session_state.prices = prices
                st.session_state.selected_tickers = selected_tickers
                st.session_state.start_date = start_date
                st.session_state.end_date = end_date

                # Menghitung pengembalian yang diharapkan
                returns = prices.pct_change().dropna() * 100  # Menghitung pengembalian dalam persen
                expected_returns = returns.apply(geometric_mean)

                # Display results
                st.write("### 📊 Returns (%)")
                st.dataframe(returns, use_container_width=True)

                # Load the BI rate data
                bi_rate_path = 'BI-7Day-RR (3).xlsx'
                bi_rate_data = pd.read_excel(bi_rate_path)

                # Clean the BI rate data
                bi_rate_data = bi_rate_data[['Tanggal', 'BI-7Day-RR']].dropna()
                bi_rate_data['Tanggal'] = pd.to_datetime(bi_rate_data['Tanggal'], errors='coerce', dayfirst=True)
                bi_rate_data['BI_7Day_RR'] = pd.to_numeric(bi_rate_data['BI-7Day-RR'], errors='coerce')
                bi_rate_data = bi_rate_data.dropna()

                # Pastikan kolom 'Tanggal' di bi_rate_data adalah tz-naive
                bi_rate_data['Tanggal'] = bi_rate_data['Tanggal'].dt.tz_localize(None)

                # Pastikan start_date dan end_date juga tz-naive
                start_date = pd.to_datetime(prices.index.min()).tz_localize(None)
                end_date = pd.to_datetime(prices.index.max()).tz_localize(None)

                # Hitung risk-free rate
                monthly_risk_free_rate = calculate_risk_free_rate(bi_rate_data, start_date, end_date)
                st.write(f"Risk-free rate: {monthly_risk_free_rate:.2f}%")

                # Filter stocks with expected returns higher than the risk-free rate
                filtered_expected_returns = expected_returns[expected_returns > monthly_risk_free_rate]

                # Prepare DataFrame for filtered stocks
                filtered_stocks = filtered_expected_returns.index.tolist()
                filtered_expected_returns_df = pd.DataFrame({
                    'Emiten': filtered_stocks,
                    'Monthly Expected Return (%)': filtered_expected_returns.values
                })

                # Display the filtered stocks
                st.subheader("📊 Emiten yang Dapat Dipertimbangkan")
                st.dataframe(filtered_expected_returns_df, use_container_width=True)

                # Ambil data return saham untuk emiten yang sudah difilter
                filtered_returns = returns[filtered_expected_returns_df['Emiten']]

                # Hitung statistik deskriptif untuk return
                descriptive_stats_returns = filtered_returns.describe().transpose()
                descriptive_stats_returns['Variance'] = filtered_returns.var()
                descriptive_stats_returns['Risk'] = filtered_returns.std()

                # Rename columns untuk kejelasan
                descriptive_stats_returns = descriptive_stats_returns.rename(columns={
                    'mean': 'Mean',
                    '50%': 'Median',
                    'min': 'Min',
                    'max': 'Max',
                })

                # Pilih kolom yang relevan
                descriptive_stats_returns = descriptive_stats_returns[['Mean', 'Variance', 'Risk', 'Median', 'Min', 'Max']]

                # Tambahkan nama emiten sebagai indeks
                descriptive_stats_returns.index.name = 'Emiten'

                # Tampilkan hasil
                st.subheader("Statistik Deskriptif untuk Return yang Difilter")
                st.dataframe(descriptive_stats_returns, use_container_width=True)

                # Menampilkan grafik pengembalian
                total_returns = filtered_returns.sum()

                # Plot total returns
                st.subheader("Total Return For Non-Eliminated Stocks")
                st.bar_chart(total_returns)

                # Portfolio Optimization
                optimization_results_df, max_sharpe_weights, cov_matrix, corr_matrix, max_sharpe_weights_df = portfolio_optimization(
                    returns, filtered_stocks, filtered_expected_returns, monthly_risk_free_rate
                )

                # Display covariance matrix
                st.subheader("Matriks Kovarian")
                st.dataframe(cov_matrix, use_container_width=True)

                # Display correlation matrix
                st.subheader("Matriks Korelasi")
                st.dataframe(corr_matrix, use_container_width=True)

                # Display optimization results
                st.subheader("Hasil Optimasi Portofolio")
                st.dataframe(optimization_results_df, use_container_width=True)

                # Display Choosen Portofolio
                st.subheader("Portofolio Efisien Terpilih Berdasarkan Sharpe Ratio Tertinggi")
                st.dataframe(max_sharpe_weights, use_container_width=True)
                
                # Display best Sharpe ratio weights
                st.subheader("Bobot Optimal Berdasarkan Sharpe Ratio Tertinggi")
                st.dataframe(max_sharpe_weights_df, use_container_width=True)

                # Plot annual returns using the highest Sharpe ratio weights
                st.subheader("Pengembalian Tahunan Portofolio")
                plot_annual_returns(filtered_returns, max_sharpe_weights_df)

                # Daftar ticker dari kolom 'Emiten' di dataframe max_sharpe_weights_df
                optimal_tickers = max_sharpe_weights_df['Emiten'].tolist()

                # Asumsikan 'prices' adalah dataframe yang berisi data harga saham bulanan yang sudah ada
                # Kamu bisa menyesuaikan dengan data yang kamu punya, misalnya harga saham bulanan
                # Download data saham berdasarkan optimal_tickers
                stock_data_df = download_stock_data(optimal_tickers, prices)

                # Menampilkan beberapa baris pertama dari data yang diunduh
                st.subheader("Harga Saham Emiten yang Tergabung Dalam Portofolio (daily)")
                st.dataframe(stock_data_df, use_container_width=True)

                stock_data_df.to_csv('stock_data.csv')

    # Form kedua untuk memilih ticker untuk analisis lebih lanjut
    if "prices" in st.session_state:
        with st.form("input_form_2"):
            st.subheader("Pilih Ticker untuk Analisis Peramalan")

            # Membaca data dari file CSV
            data = pd.read_csv('stock_data.csv', parse_dates=['Date'], index_col='Date')


            # Menghapus kolom 'Date' dari DataFrame (jika ada)
            # if 'Date' in data.columns:
            #     data = data.drop(columns=['Date'])

            # Ambil nama kolom selain 'Date' sebagai daftar ticker
            tickers = data.columns.tolist()

            # Pilih ticker untuk analisis
            selected_ticker = st.selectbox("Pilih ticker untuk analisis:", tickers)

            submit_button_2 = st.form_submit_button(label="Tampilkan Data")

            if submit_button_2:
                # Menampilkan data untuk ticker yang dipilih
                st.write(f"Data untuk {selected_ticker}:")
                st.dataframe(data[selected_ticker], use_container_width=True)

                # Plot data
                st.subheader(f"Plot Harga Saham {selected_ticker}")

                # Plot harga saham untuk ticker yang dipilih
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(data.index, data[selected_ticker], label=f'Harga Penutupan {data[selected_ticker]}')
                ax.set_title(f'Harga Penutupan {selected_ticker}')
                ax.set_xlabel('Tanggal')
                ax.set_ylabel('Harga (IDR)')
                ax.grid(True)
                st.pyplot(fig)  # Menampilkan plot di Streamlit

                #selected_data = data[selected_ticker]

                # Parameter model
                best_model_params = {
                    'Batch Size': 64,
                    'Neurons': 100,
                    'Hidden Layers': 1,
                    'Epochs': 300,
                    'Learning Rate': 0.001,
                    'Dropout': 0.2
                }

                # Jalankan model dan tampilkan hasil
                # Jalankan model
                test_mape, best_model, scaler, scaled_data = run_model(data, selected_ticker, best_model_params)

                # Tampilkan hasil MAPE
                st.write(f"Test MAPE (Denormalized): {test_mape*100:.2f}%")

                # Forecast periode ke depan
                forecast_periods = 60  # Atur jumlah periode yang akan diprediksi
                st.write(f"Forecasting untuk {forecast_periods} hari ke depan...")

                # Forecasting
                time_step = 6  # Window size yang digunakan
                forecast_results = forecast_next_periods(best_model, scaler, scaled_data, time_step, forecast_periods)

                # Tampilkan plot hasil forecast
                st.subheader("Forecasted Data")
                display_forecast_plot(scaled_data, forecast_results, forecast_periods, scaler)

                # Output hasil forecast dalam bentuk angka
                st.write("Forecasted Values (price):")
                st.dataframe(forecast_results, use_container_width=True)
                                
