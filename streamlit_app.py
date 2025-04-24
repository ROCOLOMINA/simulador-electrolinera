import streamlit as st
import pandas as pd
import numpy as np
import simpy
import random
from datetime import datetime, timedelta

# ------------------------------------------------------------------------------------
# 1. Funciones para "parsear" los DataFrames que sube el usuario
# ------------------------------------------------------------------------------------

def parse_configuration_df(config_df):
    """
    A partir de un DataFrame con columnas:
      tipo_cargador, potencia_kW, cantidad, queue_limit (opcional),
      pay_time_min, pay_time_max, pay_time_distribution (opcional)
    Devuelve:
      CHARGERS (dict), num_payment_terminals (int), PAY_TIME_MIN (float), PAY_TIME_MAX (float),
      PAY_TIME_DISTRIBUTION (str)
    """
    chargers = {}
    num_payment_terminals = 0
    pay_time_min = 5
    pay_time_max = 10
    pay_time_distribution = "uniforme"

    for _, row in config_df.iterrows():
        tipo = str(row["tipo_cargador"]).lower()
        potencia = float(row["potencia_kW"])
        cantidad = int(row["cantidad"])
        queue_limit = None
        if "queue_limit" in config_df.columns and not pd.isna(row.get("queue_limit")):
            queue_limit = int(row["queue_limit"])

        if tipo == "pago":
            num_payment_terminals = cantidad
            if "pay_time_min" in config_df.columns and not pd.isna(row.get("pay_time_min")):
                pay_time_min = float(row["pay_time_min"])
            if "pay_time_max" in config_df.columns and not pd.isna(row.get("pay_time_max")):
                pay_time_max = float(row["pay_time_max"])
            if "pay_time_distribution" in config_df.columns and not pd.isna(row.get("pay_time_distribution")):
                pay_time_distribution = row["pay_time_distribution"].strip().lower()
        else:
            chargers[row["tipo_cargador"]] = {
                "count": cantidad,
                "power": potencia,
                "queue_limit": queue_limit
            }

    return chargers, num_payment_terminals, pay_time_min, pay_time_max, pay_time_distribution


def parse_car_types_df(car_types_df):
    """
    A partir de un DataFrame con columnas:
      tipo_coche, probabilidad, bateria_media, bateria_desviacion
    Devuelve un dict CAR_TYPES.
    """
    car_types = {}
    for _, row in car_types_df.iterrows():
        car_types[row["tipo_coche"]] = {
            "probabilidad": row["probabilidad"],
            "bateria_media": row["bateria_media"],
            "bateria_desviacion": row["bateria_desviacion"],
            "count": 0
        }
    return car_types


def parse_simulation_params_df(params_df):
    """
    A partir de un DataFrame con columnas:
      tiempo_simulacion_min, numero_replicaciones
    Devuelve SIM_TIME (int) y NUM_REPLICATIONS (int).
    """
    sim_time = int(params_df.loc[0, "tiempo_simulacion_min"])
    num_reps = int(params_df.loc[0, "numero_replicaciones"])
    return sim_time, num_reps


def parse_inter_arrival_df(inter_arrival_df):
    """
    DataFrame con índices (por ejemplo: 0->Lunes, 6->Domingo) y columnas (meses).
    Devuelve el DataFrame y la lista de MONTHS.
    """
    MONTHS = list(inter_arrival_df.columns)
    return inter_arrival_df, MONTHS

# ------------------------------------------------------------------------------------
# 2. Clase Electrolinera y procesos de SimPy
# ------------------------------------------------------------------------------------

class Electrolinera:
    def __init__(self, env, CHARGERS, num_payment_terminals, PAY_TIME_MIN, PAY_TIME_MAX, PAY_TIME_DISTRIBUTION, CAR_TYPES):
        self.env = env
        self.CHARGERS = CHARGERS
        self.num_payment_terminals = num_payment_terminals
        self.PAY_TIME_MIN = PAY_TIME_MIN
        self.PAY_TIME_MAX = PAY_TIME_MAX
        self.PAY_TIME_DISTRIBUTION = PAY_TIME_DISTRIBUTION
        self.CAR_TYPES = CAR_TYPES
        self.stations = []
        # Crear los cargadores
        for charger_type, config in CHARGERS.items():
            for _ in range(config["count"]):
                self.stations.append({
                    "type": charger_type,
                    "resource": simpy.Resource(env, capacity=1),
                    "power": config["power"],
                    "queue_limit": config["queue_limit"]
                })
        # Crear los terminales de pago
        self.payment_terminal = simpy.Resource(env, capacity=self.num_payment_terminals)
        # Variables para estadísticas
        self.charger_queue_times = []
        self.payment_queue_times = []
        self.queue_lengths = []

    def charge(self, car, station, car_type, frac_carga):
        """
        Calcula el tiempo de carga con distribución normal:
          tiempo_carga (min) = frac_carga * (capacidad_coche / potencia_cargador) * 60
        """
        car_capacity = float(car_type[:-3])
        charger_power = station["power"]
        media = frac_carga * (car_capacity / charger_power) * 60
        std_dev = self.CAR_TYPES[car_type]["bateria_desviacion"] * (car_capacity / charger_power) * 60
        charging_time = max(1, np.random.normal(loc=media, scale=std_dev))
        yield self.env.timeout(charging_time)

    def pay(self, car):
        """
        Tiempo de pago según la distribución configurada.
        """
        if self.PAY_TIME_DISTRIBUTION == "normal":
            mean = (self.PAY_TIME_MIN + self.PAY_TIME_MAX) / 2
            std = (self.PAY_TIME_MAX - self.PAY_TIME_MIN) / 6
            pay_time = max(1, np.random.normal(loc=mean, scale=std))
        elif self.PAY_TIME_DISTRIBUTION == "exponencial":
            scale = (self.PAY_TIME_MIN + self.PAY_TIME_MAX) / 2
            pay_time = max(1, np.random.exponential(scale=scale))
        else:
            pay_time = random.uniform(self.PAY_TIME_MIN, self.PAY_TIME_MAX)
        yield self.env.timeout(pay_time)

# ------------------------------------------------------------------------------------
# 3. Funciones de llegada de coches y simulación con SimPy
# ------------------------------------------------------------------------------------

def get_inter_arrival_time(current_time, inter_arrival_df, MONTHS):
    month = MONTHS[current_time.month - 1]
    weekday = current_time.weekday()
    return inter_arrival_df.loc[weekday, month]


def car(env, name, electrolinera, stats):
    arrival_time = env.now
    queue_length = sum(len(station["resource"].queue) for station in electrolinera.stations)
    electrolinera.queue_lengths.append(queue_length)

    car_type = random.choices(
        list(electrolinera.CAR_TYPES.keys()),
        weights=[electrolinera.CAR_TYPES[key]["probabilidad"] for key in electrolinera.CAR_TYPES]
    )[0]
    electrolinera.CAR_TYPES[car_type]["count"] += 1

    media_bateria = electrolinera.CAR_TYPES[car_type]["bateria_media"]
    desv_bateria = electrolinera.CAR_TYPES[car_type]["bateria_desviacion"]
    frac_carga = np.random.normal(media_bateria, desv_bateria)
    frac_carga = np.clip(frac_carga, 0.01, 1.0)

    available_stations = [
        s for s in electrolinera.stations
        if s["queue_limit"] is None or len(s["resource"].queue) < s["queue_limit"]
    ]
    if not available_stations:
        stats["abandoned"] += 1
        stats["system_times"].append(0)
        return

    station = min(available_stations, key=lambda s: len(s["resource"].queue))

    with station["resource"].request() as request:
        charger_queue_start = env.now
        yield request
        charger_queue_time = env.now - charger_queue_start
        electrolinera.charger_queue_times.append(charger_queue_time)
        yield env.process(electrolinera.charge(name, station, car_type, frac_carga))

    with electrolinera.payment_terminal.request() as pay_request:
        payment_queue_start = env.now
        yield pay_request
        payment_queue_time = env.now - payment_queue_start
        electrolinera.payment_queue_times.append(payment_queue_time)
        yield env.process(electrolinera.pay(name))

    stats["served"] += 1
    stats["system_times"].append(env.now - arrival_time)


def car_generator(env, electrolinera, stats, inter_arrival_df, MONTHS, SIM_TIME):
    i = 0
    current_time = datetime(2024, 1, 1)
    while True:
        inter_arrival_time = get_inter_arrival_time(current_time, inter_arrival_df, MONTHS)
        yield env.timeout(random.expovariate(1 / inter_arrival_time))
        env.process(car(env, f"Car {i}", electrolinera, stats))
        current_time += timedelta(minutes=inter_arrival_time)
        i += 1


def run_simulation_from_dfs(config_df, car_types_df, params_df, inter_arrival_df):
    CHARGERS, num_payment_terminals, PAY_TIME_MIN, PAY_TIME_MAX, PAY_TIME_DISTRIBUTION = parse_configuration_df(config_df)
    CAR_TYPES = parse_car_types_df(car_types_df)
    SIM_TIME, NUM_REPLICATIONS = parse_simulation_params_df(params_df)
    inter_arrival_parsed, MONTHS = parse_inter_arrival_df(inter_arrival_df)

    def run_simulation(replication):
        random.seed(42 + replication)
        np.random.seed(42 + replication)
        env = simpy.Environment()
        electrolinera = Electrolinera(
            env,
            CHARGERS,
            num_payment_terminals,
            PAY_TIME_MIN,
            PAY_TIME_MAX,
            PAY_TIME_DISTRIBUTION,
            CAR_TYPES
        )
        stats = {"served": 0, "abandoned": 0, "system_times": []}
        env.process(car_generator(env, electrolinera, stats, inter_arrival_parsed, MONTHS, SIM_TIME))
        env.run(until=SIM_TIME)

        # Cálculo de métricas base
        car_type_counts = {key: electrolinera.CAR_TYPES[key]["count"] for key in electrolinera.CAR_TYPES}
        results = {
            "Coches atendidos": stats["served"],
            "Coches abandonados": stats["abandoned"],
            "Tasa de abandono (%)": (stats["abandoned"] / (stats["served"] + stats["abandoned"]) * 100) if (stats["served"] + stats["abandoned"]) > 0 else 0,
            "Tiempo promedio en el sistema (min)": np.mean(stats["system_times"]) if stats["system_times"] else 0,
            "Tiempo máximo en el sistema (min)": max(stats["system_times"], default=0),
            "Tiempo mínimo en el sistema (min)": min(stats["system_times"], default=0),
            "Tiempo promedio en cola de cargadores (min)": (np.mean(electrolinera.charger_queue_times) if electrolinera.charger_queue_times else 0),
            "Tiempo máximo en cola de cargadores (min)": max(electrolinera.charger_queue_times, default=0),
            "Tiempo mínimo en cola de cargadores (min)": min(electrolinera.charger_queue_times, default=0),
            "Tiempo promedio en cola de pago (min)": (np.mean(electrolinera.payment_queue_times) if electrolinera.payment_queue_times else 0),
            "Tiempo máximo en cola de pago (min)": max(electrolinera.payment_queue_times, default=0),
            "Tiempo mínimo en cola de pago (min)": min(electrolinera.payment_queue_times, default=0),
            "Longitud promedio de la cola": (np.mean(electrolinera.queue_lengths) if electrolinera.queue_lengths else 0),
            "Longitud máxima de la cola": max(electrolinera.queue_lengths, default=0)
        }
        # Métricas adicionales solicitadas
        total_charger_wait = np.array(electrolinera.charger_queue_times)
        total_payment_wait = np.array(electrolinera.payment_queue_times)
        waits = total_charger_wait + total_payment_wait
        served = stats["served"]
        # Tiempo medio de espera por coche
        results["Tiempo medio de espera por coche (min)"] = np.mean(waits) if waits.size > 0 else 0
        # Porcentaje de coches que tienen que esperar
        results["Porcentaje de coches que esperan (%)"] = (np.sum(total_charger_wait > 0) / served * 100) if served > 0 else 0
        # Tasa de ocupación de los cargadores
        total_busy = sum(s["busy_time"] for s in electrolinera.stations)
        num_chargers = len(electrolinera.stations)
        results["Tasa de ocupación cargadores (%)"] = (total_busy / (num_chargers * SIM_TIME) * 100) if num_chargers > 0 else 0
        # Tiempo medio de carga
        results["Tiempo medio de carga (min)"] = np.mean(electrolinera.charging_times) if electrolinera.charging_times else 0
        # Carga media suministrada por cargador por hora
        total_energy = sum(s["energy_delivered"] for s in electrolinera.stations)
        hours = SIM_TIME / 60
        results["Carga media suministrada por cargador por hora (kWh/h)"] = (total_energy / num_chargers / hours) if num_chargers > 0 and hours > 0 else 0
        # Número de coches atendidos por día
        days = SIM_TIME / 1440
        results["Coches atendidos por día"] = served / days if days > 0 else 0
        # Energía total suministrada por tipo de cargador
        energy_by_type = {}
        for s in electrolinera.stations:
            energy_by_type.setdefault(s["type"], 0)
            energy_by_type[s["type"]] += s["energy_delivered"]
        for tipo, energia in energy_by_type.items():
            results[f"Energía total suministrada {tipo} (kWh)"] = energia
        # Añadir conteo por tipo de coche
        results.update(car_type_counts)
        return results

    results = []
    for rep in range(NUM_REPLICATIONS):
        results.append(run_simulation(rep))
    df_results = pd.DataFrame(results)
    return df_results

# ------------------------------------------------------------------------------------
# 4. Interfaz de usuario con Streamlit
# ------------------------------------------------------------------------------------

def main():
    st.title("Simulador de Electrolinera")
    st.write("Bienvenido/a a la aplicación de simulación. Sube tus CSV o utiliza los datos de ejemplo.")

    st.subheader("1) Configuración de Cargadores")
    config_file = st.file_uploader("Sube 'configuracion_cargadores.csv'", type=["csv"])
    if config_file is not None:
        config_df = pd.read_csv(config_file)
        st.dataframe(config_df)
    else:
        st.info("Usando configuración de ejemplo.")
        config_df = pd.DataFrame({
            "tipo_cargador": ["100kW", "350kW", "pago"],
            "potencia_kW": [100, 350, 0],
            "cantidad": [1, 1, 1],
            "queue_limit": [1, 1, None],
            "pay_time_min": [None, None, 5],
            "pay_time_max": [None, None, 10],
            "pay_time_distribution": [None, None, "uniforme"]
        })
        st.dataframe(config_df)

    st.subheader("2) Tipos de Coches")
    cars_file = st.file_uploader("Sube 'tipos_coches.csv'", type=["csv"])
    if cars_file is not None:
        car_types_df = pd.read_csv(cars_file)
        st.dataframe(car_types_df)
    else:
        st.info("Usando tipos de coches de ejemplo.")
        car_types_df = pd.DataFrame({
            "tipo_coche": ["120kWh", "70kWh", "40kWh"],
            "probabilidad": [0.2, 0.3, 0.5],
            "bateria_media": [0.6, 0.6, 0.6],
            "bateria_desviacion": [0.2, 0.2, 0.2]
        })
        st.dataframe(car_types_df)

    st.subheader("3) Parámetros de Simulación")
    params_file = st.file_uploader("Sube 'parametros_simulacion.csv'", type=["csv"])
    if params_file is not None:
        params_df = pd.read_csv(params_file)
        st.dataframe(params_df)
    else:
        st.info("Usando parámetros de ejemplo.")
        params_df = pd.DataFrame({
            "tiempo_simulacion_min": [525600],
            "numero_replicaciones": [1]
        })
        st.dataframe(params_df)

    st.subheader("4) Tiempos de Llegada (inter_arrival_times.csv)")
    inter_file = st.file_uploader("Sube 'inter_arrival_times.csv'", type=["csv"])
    if inter_file is not None:
        inter_arrival_df = pd.read_csv(inter_file, index_col=0)
        st.dataframe(inter_arrival_df)
    else:
        st.info("Usando tiempos de llegada de ejemplo.")
        inter_arrival_df = pd.DataFrame({
            'January': [10.015, 10.009, 9.925, 10.442, 7.793, 12.118, 14.094],
            'February': [8.648, 9.637, 9.383, 8.274, 6.288, 10.572, 12.811],
            'March': [10.657, 10.127, 10.110, 9.292, 6.941, 11.133, 13.279],
            'April': [8.498, 8.365, 7.140, 6.828, 6.645, 10.373, 10.516],
            'May': [9.786, 9.091, 9.034, 8.031, 5.786, 9.334, 11.812],
            'June': [8.650, 8.987, 8.570, 7.637, 5.632, 8.884, 11.082],
            'July': [8.067, 8.154, 8.086, 6.829, 5.302, 6.786, 9.329],
            'August': [7.038, 7.992, 8.115, 7.495, 5.710, 7.907, 10.169],
            'September': [9.417, 9.227, 9.118, 8.120, 6.035, 9.613, 11.541],
            'October': [9.249, 9.180, 9.063, 8.385, 5.974, 8.763, 8.712],
            'November': [9.562, 9.519, 9.508, 8.961, 6.735, 10.611, 11.845],
            'December': [9.244, 9.089, 8.677, 8.355, 6.241, 9.412, 10.608]
        }, index=[0,1,2,3,4,5,6])
        st.dataframe(inter_arrival_df)

    if st.button("Ejecutar Simulación"):
        st.write("Ejecutando simulación, por favor espera...")
        results_df = run_simulation_from_dfs(config_df, car_types_df, params_df, inter_arrival_df)
        st.success("Simulación finalizada.")
        st.subheader("Resultados")
        st.dataframe(results_df)
        csv_result = results_df.to_csv(index=False)
        st.download_button(
            label="Descargar resultados",
            data=csv_result,
            file_name="resultados_simulacion.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
