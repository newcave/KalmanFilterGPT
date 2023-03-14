import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

def get_water(variance):
    """Measuring Data."""
    # variance = 2
    v = np.random.normal(0, variance)
    water_true = 100.0
    z_water_meas = water_true + v
    return z_water_meas

def kalman_filter(z_meas, x_esti, P):
    """Kalman Filter Algorithm for One Variable."""
    A = 1
    H = 1
    Q = 0
    R = 4
    
    # (1) Prediction.
    x_pred = A * x_esti
    P_pred = A * P * A + Q

    # (2) Kalman Gain.
    K = P_pred * H / (H * P_pred * H + R)

    # (3) Estimation.
    x_esti = x_pred + K * (z_meas - H * x_pred)

    # (4) Error Covariance.
    P = P_pred - K * H * P_pred

    return x_esti, P

# Define Streamlit app.
def app():
   
#   st.beta_set_page_config(page_title="Kalman Filter", page_icon=":bar_chart:", layout="wide", initial_sidebar_state="expanded")
    st.sidebar.image("logo-ailab.png", use_column_width=True)
    st.sidebar.markdown("# Kalman Filter Setting")
    st.title("칼만필터학습기 by KSH x ChatGPT")
 
    sidebar = st.sidebar
    time_end = sidebar.slider("Time end (hrs.) [분석할 시간]", min_value=1, max_value=48, value=10, step=1)
    dt = sidebar.slider("Time step (hrs.) [분석할 빈도]", min_value=0.1, max_value=1.0, value=0.4, step=0.1)
    x_0 = sidebar.slider("Initial Streamflow estimate [초기 유량 추정치]", min_value=0.0, max_value=1000.0, value=110.0, step=10.0)
    P_0 = sidebar.slider("Init. err. covariance estimate [초기 공분산 추정치, 시스템을 잘 모르면 큰 값]", min_value=0.0, max_value=20.0, value=9.0, step=0.5)
    variance = sidebar.slider("Standard Deviation of Data(for making) [가상의 측정자료의 표준편차]", min_value=0.0, max_value=10.0, value=2.0, step=1.0)

    time = np.arange(0, time_end, dt)
    n_samples = len(time)
    water_meas_save = np.zeros(n_samples)
    water_esti_save = np.zeros(n_samples)

    x_esti, P = None, None
    for i in range(n_samples):
        z_meas = get_water(variance)
        if i == 0:
            x_esti, P = x_0, P_0
        else:
            x_esti, P = kalman_filter(z_meas, x_esti, P)
        water_meas_save[i] = z_meas
        water_esti_save[i] = x_esti

    fig, ax = plt.subplots()
    ax.plot(time, water_meas_save, 'r*--', label='Measurements')
    ax.plot(time, water_esti_save, 'bo-', label='Kalman Filter')
    ax.legend(loc='upper left')
    ax.set_title('Measurements versus Estimation (KF method)')
    ax.set_xlabel('Time [hrs.]')
    ax.set_ylabel('Streamflow [CMS]')
    st.pyplot(fig)

if __name__ == '__main__':
    app()
