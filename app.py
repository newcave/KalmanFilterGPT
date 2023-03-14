import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

def get_water():
    """Measuring Data."""
    v = np.random.normal(0, 2)
    water_true = 14.4
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
    
    st.set_page_config(page_title="Kalman Filter Example", page_icon=":bar_chart:", layout="wide")
    st.sidebar.image("logo-ailab.png", use_column_width=True)
    st.sidebar.title("Kalman Filter Settings")
    st.title("Kalman Filter Example")
 
    sidebar = st.sidebar
    time_end = sidebar.slider("Time end (seconds)", min_value=1, max_value=20, value=10, step=1)
    dt = sidebar.slider("Time step (seconds)", min_value=0.1, max_value=1.0, value=0.2, step=0.1)
    x_0 = sidebar.slider("Initial Streamflow estimate", min_value=0.0, max_value=20.0, value=12.0, step=0.1)
    P_0 = sidebar.slider("Initial error covariance estimate", min_value=0.0, max_value=20.0, value=6.0, step=0.1)

    time = np.arange(0, time_end, dt)
    n_samples = len(time)
    water_meas_save = np.zeros(n_samples)
    water_esti_save = np.zeros(n_samples)

    x_esti, P = None, None
    for i in range(n_samples):
        z_meas = get_water()
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
    ax.set_title('Measurements v.s. Estimation (Kalman Filter)')
    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Streamlow [Q-ND]')
    st.pyplot(fig)

if __name__ == '__main__':
    app()
