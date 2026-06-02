import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

def rocket_analysis_actual(
        t, x, T1, T2, m01, m02, rme1, rme2, mp1, mp2,
        g0, RE, Y_ini, h0, CD, p0, A, hTower):

    # State variables
    v = x[0]
    gamma = x[1]
    h = x[2]
    downrange = x[3]

    tburn1 = mp1 / rme1
    tburn2 = mp2 / rme2
    tf = tburn1 + tburn2

    # Stage selection
    if t <= tburn1:
        m0 = m01
        T = T1
        rme = rme1
        m = m0 - rme * t
    elif t <= tf:
        m0 = m02
        T = T2
        rme = rme2
        # Stage-2 burn starts at t = tburn1
        m = m0 - rme * (t - tburn1)
    else:
        T = 0.0
        m = m02 - mp2

    # Prevent divide-by-zero
    m = max(m, 1.0)

    # Gravity variation
    gc = g0 / (1.0 + h / RE)**2

    # Atmospheric density
    rho = p0 * np.exp(-h / h0)

    # Dynamic pressure
    q = 0.5 * rho * v**2

    drag = q * A * CD

    f_dot = np.zeros(6)

    # Before tower clearance
    if h < hTower:
        f_dot[0] = (T - drag - m * gc * np.sin(Y_ini)) / m
        f_dot[1] = 0.0
        f_dot[2] = v * np.sin(Y_ini)
        f_dot[3] = (RE / (RE + h)) * v * np.cos(Y_ini)
        f_dot[4] = drag / m
        f_dot[5] = gc * np.sin(Y_ini)
    else:
        if abs(v) < 1e-6:
            gamma_dot = 0.0
        else:
            gamma_dot = (-(1.0 / v) * (gc - v**2 / (RE + h)) * np.cos(gamma))

        f_dot[0] = (T - drag - m * gc * np.sin(gamma)) / m
        f_dot[1] = gamma_dot
        f_dot[2] = v * np.sin(gamma)
        f_dot[3] = (RE / (RE + h)) * v * np.cos(gamma)
        f_dot[4] = drag / m
        f_dot[5] = gc * np.sin(gamma)

    return f_dot

def get_base64_image():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_b64

def run_simulation(m01=120000.0, n=5.0, Isp1=320.0, Isp2=350.0, T2W1=1.4, T2W2=1.5, CD=0.5, D=5.0, hTower=110.0):
    g0 = 9.81
    c1 = Isp1 * g0
    c2 = Isp2 * g0
    p0 = 1.225
    h0 = 7.5 * 10**3
    Y_ini = 89.75 * np.pi / 180
    RE = 6378000.0
    A = np.pi * (D / 2.0)**2

    mf1 = m01 / n
    m02 = mf1
    mf2 = m02 / n
    T1 = T2W1 * m01 * g0
    T2 = T2W2 * m02 * g0
    rme1 = T1 / c1
    rme2 = T2 / c2
    mp1 = m01 - mf1
    mp2 = m02 - mf2
    tburn1 = mp1 / rme1
    tburn2 = mp2 / rme2
    t0 = 0.0
    tf = tburn1 + tburn2

    init_params = [0.0, Y_ini, 0.0, 0.0, 0.0, 0.0]

    sol = solve_ivp(
        rocket_analysis_actual,
        [t0, tf],
        init_params,
        args=(T1, T2, m01, m02, rme1, rme2, mp1, mp2, g0, RE, Y_ini, h0, CD, p0, A, hTower),
        rtol=1e-4
    )

    t = sol.t
    X = sol.y.T
    v = X[:, 0]
    Y = X[:, 1]
    h = X[:, 2]
    x_range = X[:, 3]
    VelLostDrag = X[:, 4]
    VelLostGravity = X[:, 5]
    Y_a_deg = np.degrees(Y)

    density = p0 * np.exp(-h / h0)
    dynamicpressure = 0.5 * density * v**2

    j = 100000.0          # Karman line height (m)
    escapevel = 11200.0  # Escape velocity (m/s)
    timek = 167.0

    Index_maxdynamicpressure = np.argmax(dynamicpressure)
    dynamicpressuremax = dynamicpressure[Index_maxdynamicpressure]
    HeightMaxPos = h[Index_maxdynamicpressure]

    ideal_dv = c1 * np.log(n) + c2 * np.log(n)
    vel_eff = 100 * v[-1] / ideal_dv if ideal_dv > 0 else 0

    results = {
        'max_q': float(dynamicpressuremax),
        'h_max_q': float(HeightMaxPos),
        'tburn1': float(tburn1),
        'tburn2': float(tburn2),
        'tf': float(tf),
        'v_final': float(v[-1]),
        'h_final': float(h[-1]),
        'gamma_final': float(Y_a_deg[-1]),
        'range_final': float(x_range[-1]),
        'drag_loss': float(VelLostDrag[-1]),
        'grav_loss': float(VelLostGravity[-1]),
        'total_loss': float(VelLostDrag[-1] + VelLostGravity[-1]),
        'ideal_dv': float(ideal_dv),
        'vel_eff': float(vel_eff)
    }

    charts = {}

    # Figure 1: Velocity
    plt.figure()
    plt.plot(t, v, 'r-', label='Velocity')
    plt.axhline(escapevel, color='magenta', label='Escape Velocity')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Velocity vs Time')
    plt.legend()
    plt.grid(True)
    charts['chart_vel'] = get_base64_image()

    # Figure 2: Altitude
    plt.figure()
    plt.plot(t, h, 'b-', label='Altitude')
    plt.axhline(j, color='red', label='Karman Line')
    plt.xlabel('Time (s)')
    plt.ylabel('Altitude (m)')
    plt.title('Altitude vs Time')
    plt.legend()
    plt.grid(True)
    charts['chart_alt'] = get_base64_image()

    # Figure 3: Dynamic Pressure
    plt.figure()
    plt.plot(t, dynamicpressure, 'r-')
    plt.axhline(dynamicpressuremax, color='green')
    plt.xlabel('Time (s)')
    plt.ylabel('Dynamic Pressure (Pa)')
    plt.title('Dynamic Pressure vs Time')
    plt.grid(True)
    charts['chart_q'] = get_base64_image()
    
    # Figure 4: Flight Path
    plt.figure()
    plt.plot(t, Y_a_deg, 'm-')
    plt.xlabel('Time (s)')
    plt.ylabel('Flight Path Angle (deg)')
    plt.title('Flight Path Angle vs Time')
    plt.grid(True)
    charts['chart_gamma'] = get_base64_image()

    return results, charts
