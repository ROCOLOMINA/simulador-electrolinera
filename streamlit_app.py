import streamlit as st
import pandas as pd
import numpy as np
import simpy
import random
from datetime import datetime, timedelta

# ------------------------------------------------------------------------------------
# 1. Funciones para "parsear" los DataFrames que sube el usuario
#    en lugar de leer archivos CSV directamente.
# ------------------------------------------------------------------------------------

def parse_configuration_df(config_df):
    """
    A partir de un DataFrame con columnas:
      tipo_cargador, potencia_kW, cantidad, queue_limit (opcional),
      pay_time_min, pay_time_max (opcional si tipo_cargador=='pago')
    Devuelve:
      CHARGERS (dict), num_payment_terminals (int), PAY_TIME_MIN (float), PAY_TIME_MAX (float)
    """
    chargers = {}
    num_payment_terminals = 0
    pay_time_min = 5
    pay_time_max = 10

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
        else:
            chargers[row["tipo_cargador"]] = {
                "count": cantidad,
                "power": potencia,
                "queue_limit": queue_limit
            }

    return chargers, num_payment_terminals, pay_time_min, pay_time_max


def parse_car_types_df(car_types_df):
    """
    A partir de un DataFrame con columnas:
      tipo_coche, probabilidad, bateria_media, bateria_desviacion
    Devuelve un dict CAR_TYPES.
    """
    car_types = {}
    for _, row in car_types_df.iterrows():
        car_types[row["tipo_coche"]] = {
            "probability": row["probabilidad"],
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
    DataFrame con índices [Lunes,Martes,...] y columnas [Enero, Febrero, ...].
    Devuelve la matriz de tiempos y la lista de MONTHS para usarlos en get_inter_arrival_time.
    """
    # Simplemente retornamos el DF tal cual y la lista de columnas
    MONTHS = list(inter_arrival_df.columns)
    return inter_arrival_df, MONTHS

# ------------------------------------------------------------------------------------
# 2. Clase Electrolinera y procesos de SimPy
# ------------------------------------------------------------------------------------

class Electrolinera:
    def __init__(self, env, CHARGERS, num_payment_terminals, PAY_TIME_MIN, PAY_TIME_MAX, CAR_TYPES):
        self.env = env
        self.CHARGERS = CHARGERS
        self.num_payment_terminals = num_payment_terminals
        self.PAY_TIME_MIN = PAY_TIME_MIN
        self.PAY_TIME_MAX = PAY_TIME_MAX
        self.CAR_TYPES = CAR_TYPES

        self.stations = []
        # Crear los cargadores
        for charger_type, config in CHARGERS.items():
            for _ in range(config["count"]):
                self.stations.append(
                    {
                        "type": charger_type,
                        "resource": simpy.Resource(env, capacity=1),
                        "power": config["power"],
                        "queue_limit": config["queue_limit"]
                    }
                )
        # Crear los terminales de pago
        self.payment_terminal = simpy.Resource(env, capacity=self.num_payment_terminals)

        # Variables para estadísticas
        self.charger_queue_times = []
        self.payment_queue_times = []
        self.queue_lengths = []

        # Lo guardamos para usarlos dentro de charge()
        self.CAR_TYPES = CAR_TYPES

    def charge(self, car, station, car_type, frac_carga):
        """
        Calcula el tiempo de carga con distribución normal:
            tiempo_carga (min) = frac_carga * (capacidad_coche / potencia_cargador) * 60
        """
        # convertir "50 kWh" => 50.0 (capacidad del coche)
        car_capacity = float(car_type[:-3])
        charger_power = station["power"]

        media = frac_carga * (car_capacity / charger_power) * 60
        std_dev = self.CAR_TYPES[car_type]["bateria_desviacion"] * (car_capacity / charger_power) * 60

        charging_time = max(1, np.random.normal(loc=media, scale=std_dev))
        yield self.env.timeout(charging_time)

    def pay(self, car):
        """
        Tiempo de pago uniforme entre PAY_TIME_MIN y PAY_TIME_MAX
        """
        pay_time = random.uniform(self.PAY_TIME_MIN, self.PAY_TIME_MAX)
        yield self.env.timeout(pay_time)

# ------------------------------------------------------------------------------------
# 3. Funciones de llegada de coches
# ------------------------------------------------------------------------------------

def get_inter_arrival_time(current_time, inter_arrival_df, MONTHS):
    """
    Devuelve el tiempo medio de llegada en minutos, según la fecha (mes, día de la semana).
    """
    month = MONTHS[current_time.month - 1]
    weekday = current_time.weekday()  # 0=lunes, 6=domingo
    return inter_arrival_df.loc[weekday, month]

def car(env, name, electrolinera, stats):
    """
    Proceso de cada coche que llega:
    """
    arrival_time = env.now
    queue_length = sum(len(station["resource"].queue) for station in electrolinera.stations)
    electrolinera.queue_lengths.append(queue_length)

    # Escoger tipo de coche según probabilidad
    car_type = random.choices(
        list(electrolinera.CAR_TYPES.keys()),
        weights=[electrolinera.CAR_TYPES[key]["probability"] for key in electrolinera.CAR_TYPES]
    )[0]
    electrolinera.CAR_TYPES[car_type]["count"] += 1

    # Generar fracción de batería que necesita cargar (entre 0.01 y 1.0)
    media_bateria = electrolinera.CAR_TYPES[car_type]["bateria_media"]
    desv_bateria = electrolinera.CAR_TYPES[car_type]["bateria_desviacion"]
    frac_carga = np.random.normal(media_bateria, desv_bateria)
    frac_carga = np.clip(frac_carga, 0.01, 1.0)

    # Filtrar estaciones con hueco
    available_stations = [
        s for s in electrolinera.stations
        if s["queue_limit"] is None or len(s["resource"].queue) < s["queue_limit"]
    ]
    if not available_stations:
        # Abandono
        stats["abandoned"] += 1
        stats["system_times"].append(0)
        return

    # Elegir la estación con la cola más corta
    station = min(available_stations, key=lambda s: len(s["resource"].queue))

    # Cola de cargador
    with station["resource"].request() as request:
        charger_queue_start = env.now
        yield request
        charger_queue_time = env.now - charger_queue_start
        electrolinera.charger_queue_times.append(charger_queue_time)

        # Proceso de carga
        yield env.process(electrolinera.charge(name, station, car_type, frac_carga))

    # Cola de pago
    with electrolinera.payment_terminal.request() as pay_request:
        payment_queue_start = env.now
        yield pay_request
        payment_queue_time = env.now - payment_queue_start
        electrolinera.payment_queue_times.append(payment_queue_time)

        # Proceso de pago
        yield env.process(electrolinera.pay(name))

    # Estadísticas finales
    stats["served"] += 1
    stats["system_times"].append(env.now - arrival_time)

def car_generator(env, electrolinera, stats, inter_arrival_df, MONTHS, SIM_TIME):
    """
    Genera coches de forma continua, usando la tabla inter_arrival_df.
    """
    i = 0
    current_time = datetime(2024, 1, 1)
    while True:
        inter_arrival_time = get_inter_arrival_time(current_time, inter_arrival_df, MONTHS)
        yield env.timeout(random.expovariate(1 / inter_arrival_time))

        env.process(car(env, f"Car {i}", electrolinera, stats))
        current_time += timedelta(minutes=inter_arrival_time)
        i += 1

# ------------------------------------------------------------------------------------
# 4. Función principal para correr la simulación *desde DataFrames*
# ------------------------------------------------------------------------------------

def run_simulation_from_dfs(config_df, car_types_df, params_df, inter_arrival_df):
    """
    Esta función toma los DataFrames subidos por el usuario (o por defecto),
    recrea la lógica que antes estaba en CSV, y ejecuta la simulación.

    Devuelve un DataFrame con los resultados agregados de todas las replicaciones.
    """
    # 1) Parsear DataFrames
    CHARGERS, num_payment_terminals, PAY_TIME_MIN, PAY_TIME_MAX = parse_configuration_df(config_df)
    CAR_TYPES = parse_car_types_df(car_types_df)
    SIM_TIME, NUM_REPLICATIONS = parse_simulation_params_df(params_df)
    inter_arrival_parsed, MONTHS = parse_inter_arrival_df(inter_arrival_df)

    # 2) Definir una función interna para correr UNA réplica
    def run_simulation(replication):
        # Fijar semilla reproducible
        random.seed(42 + replication)
        np.random.seed(42 + replication)

        # Crear entorno y electrolinera
        env = simpy.Environment()
        electrolinera = Electrolinera(
            env,
            CHARGERS,
            num_payment_terminals,
            PAY_TIME_MIN,
            PAY_TIME_MAX,
            CAR_TYPES
        )

        # Stats
        stats = {"served": 0, "abandoned": 0, "system_times": []}

        # Iniciar generación de coches
        env.process(
            car_generator(env, electrolinera, stats, inter_arrival_parsed, MONTHS, SIM_TIME)
        )
        env.run(until=SIM_TIME)

        # Recoger estadísticos
        car_type_counts = {key: electrolinera.CAR_TYPES[key]["count"] for key in electrolinera.CAR_TYPES}

        return {
            "Coches atendidos": stats["served"],
            "Coches abandonados": stats["abandoned"],
            "Tasa de abandono (%)": (
                stats["abandoned"] / (stats["served"] + stats["abandoned"]) * 100
            ) if (stats["served"] + stats["abandoned"]) > 0 else 0,
            "Tiempo promedio en el sistema (min)": np.mean(stats["system_times"]) if stats["system_times"] else 0,
            "Tiempo máximo en el sistema (min)": max(stats["system_times"], default=0),
            "Tiempo mínimo en el sistema (min)": min(stats["system_times"], default=0),
            "Tiempo promedio en cola de cargadores (min)": (
                np.mean(electrolinera.charger_queue_times) if electrolinera.charger_queue_times else 0
            ),
            "Tiempo máximo en cola de cargadores (min)": max(electrolinera.charger_queue_times, default=0),
            "Tiempo mínimo en cola de cargadores (min)": min(electrolinera.charger_queue_times, default=0),
            "Tiempo promedio en cola de pago (min)": (
                np.mean(electrolinera.payment_queue_times) if electrolinera.payment_queue_times else 0
            ),
            "Tiempo máximo en cola de pago (min)": max(electrolinera.payment_queue_times, default=0),
            "Tiempo mínimo en cola de pago (min)": min(electrolinera.payment_queue_times, default=0),
            "Longitud promedio de la cola": (
                np.mean(electrolinera.queue_lengths) if electrolinera.queue_lengths else 0
            ),
            "Longitud máxima de la cola": max(electrolinera.queue_lengths, default=0),
            **car_type_counts
        }

    # 3) Ejecutar tantas réplicas como indique NUM_REPLICATIONS
    results = []
    for rep in range(NUM_REPLICATIONS):
        results.append(run_simulation(rep))

    # 4) Unir y devolver resultados como DataFrame
    df_results = pd.DataFrame(results)
    return df_results

# ------------------------------------------------------------------------------------
# 5. Interfaz de Streamlit
# ------------------------------------------------------------------------------------

def main():
    st.title("Simulador de Electrolinera")
    st.write(
        "Bienvenido/a a la aplicación de simulación con Streamlit. "
        "Sube tus CSV o utiliza los valores por defecto."
    )

    # -----------------------------------------------------------
    # 1) Subir o no el CSV de configuración
    # -----------------------------------------------------------
    st.subheader("1) Configuración de Cargadores")
    config_file = st.file_uploader("Sube 'configuracion_cargadores.csv'", type=["csv"])
    if config_file is not None:
        config_df = pd.read_csv(config_file)
        st.write("Vista previa:")
        st.dataframe(config_df)
    else:
        st.info("Usando configuración de ejemplo.")
        config_df = pd.DataFrame({
            "tipo_cargador": ["rapido", "pago"],
            "potencia_kW": [50, 0],
            "cantidad": [2, 1],
            "queue_limit": [None, None]
            # si deseas pay_time_min, pay_time_max, puedes añadir columnas
        })
        st.dataframe(config_df)

    # -----------------------------------------------------------
    # 2) Subir o no el CSV de tipos de coches
    # -----------------------------------------------------------
    st.subheader("2) Tipos de Coches")
    cars_file = st.file_uploader("Sube 'tipos_coches.csv'", type=["csv"])
    if cars_file is not None:
        car_types_df = pd.read_csv(cars_file)
        st.dataframe(car_types_df)
    else:
        st.info("Usando tipos de coches de ejemplo.")
        car_types_df = pd.DataFrame({
            "tipo_coche": ["50 kWh", "70 kWh"],
            "probabilidad": [0.6, 0.4],
            "bateria_media": [0.3, 0.4],
            "bateria_desviacion": [0.05, 0.1]
        })
        st.dataframe(car_types_df)

    # -----------------------------------------------------------
    # 3) Parámetros de simulación
    # -----------------------------------------------------------
    st.subheader("3) Parámetros de Simulación")
    params_file = st.file_uploader("Sube 'parametros_simulacion.csv'", type=["csv"])
    if params_file is not None:
        params_df = pd.read_csv(params_file)
        st.dataframe(params_df)
    else:
        st.info("Usando parámetros por defecto.")
        params_df = pd.DataFrame({
            "tiempo_simulacion_min": [1440],
            "numero_replicaciones": [1]
        })
        st.dataframe(params_df)

    # -----------------------------------------------------------
    # 4) Tiempos de llegada
    # -----------------------------------------------------------
    st.subheader("4) Tiempos de Llegada (inter_arrival_times.csv)")
    inter_file = st.file_uploader("Sube 'inter_arrival_times.csv'", type=["csv"])
    if inter_file is not None:
        inter_arrival_df = pd.read_csv(inter_file, index_col=0)
        st.dataframe(inter_arrival_df)
    else:
        st.info("Usando tiempos de llegada de ejemplo.")
        inter_arrival_df = pd.DataFrame({
            "Enero": [5, 5, 5, 5, 5, 5, 5],
        }, index=[0,1,2,3,4,5,6])  # 0->Lunes, 6->Domingo
        st.dataframe(inter_arrival_df)

    # -----------------------------------------------------------
    # Botón para ejecutar
    # -----------------------------------------------------------
    if st.button("Ejecutar Simulación"):
        st.write("Corriendo la simulación, por favor espera...")
        results_df = run_simulation_from_dfs(config_df, car_types_df, params_df, inter_arrival_df)
        st.success("Simulación finalizada.")
        st.subheader("Resultados")
        st.dataframe(results_df)

        # Botón para descargar CSV
        csv_result = results_df.to_csv(index=False)
        st.download_button(
            label="Descargar resultados",
            data=csv_result,
            file_name="resultados_simulacion.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
