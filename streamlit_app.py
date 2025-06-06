import streamlit as st
import pandas as pd
import numpy as np
import simpy
import random
from datetime import datetime, timedelta

# ------------------------------------------------------------------------------------
# Diccionarios de traducción para días y meses en español
# ------------------------------------------------------------------------------------

DIAS_SEMANA = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
MESES = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
         "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]


# ------------------------------------------------------------------------------------
# 1. Funciones para "parsear" los DataFrames que sube el usuario
# ------------------------------------------------------------------------------------

def parse_configuration_df(config_df):
    """
    A partir de un DataFrame con columnas:
      tipo_cargador, potencia_kW, cantidad, queue_limit (opcional)
    Devuelve:
      CHARGERS (dict)
    """
    chargers = {}

    for _, row in config_df.iterrows():
        tipo = row["tipo_cargador"]
        potencia = float(row["potencia_kW"])
        cantidad = int(row["cantidad"])

        # extraemos queue_limit solo si existe y no es NaN
        queue_limit = None
        if "limite_cola" in config_df.columns and not pd.isna(row["limite_cola"]):
            queue_limit = int(row["limite_cola"])

        # añadimos directamente
        chargers[tipo] = {
            "count": cantidad,
            "power": potencia,
            "queue_limit": queue_limit,
            "precio_venta": float(row["precio_venta_euros/kWh"]) if "precio_venta_euros/kWh" in config_df.columns else 0.5
        }

    return chargers

def parse_car_types_df(car_types_df):
    """
    A partir de un DataFrame con columnas:
      tipo_VE, probabilidad, bateria_media, bateria_desviacion
    Devuelve un dict CAR_TYPES.
    """
    car_types = {}
    for _, row in car_types_df.iterrows():
        car_types[row["tipo_VE"]] = {
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
    start_date = datetime.strptime(params_df.loc[0, "fecha_inicio_simulacion"], "%Y-%m-%d")
    return sim_time, num_reps, start_date

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
    def __init__(self, env, CHARGERS, CAR_TYPES):
        self.env = env
        self.CHARGERS = CHARGERS
        self.CAR_TYPES = CAR_TYPES

        # Estadísticas nuevas
        self.total_wait_time = 0.0                   # sumatorio de todo el tiempo de espera (antes de cargar)
        self.waiting_cars = 0                        # cuántos coches han tenido que esperar
        self.charging_times = []                     # lista de tiempos de carga individuales
        self.energy_by_type = {                      # energía kWh entregada, por tipo de cargador
            ctype: 0.0
            for ctype in CHARGERS.keys()
        }

        self.stats_by_charger_type = {
        ctype: {
            "queue_times": [],
            "charging_times": [],
            "queue_lengths": [],
            "busy_time": 0.0,
            "served": 0
        }
        for ctype in CHARGERS
        }


        self.stations = []
        # Crear los cargadores
        for charger_type, config in CHARGERS.items():
            for _ in range(config["count"]):
                self.stations.append({
                    "type": charger_type,
                    "resource": simpy.Resource(env, capacity=1),
                    "power": config["power"],
                    "queue_limit": config["queue_limit"],
                    "busy_time": 0.0
                })

        # Variables para estadísticas
        self.charger_queue_times = []
        self.queue_lengths = []

    def charge(self, car, station, car_type, frac_carga):
        car_capacity = float(car_type[:-3])     # kWh
        charger_power = station["power"]        # kW

        # distribuye tiempo (minutos)
        mean = frac_carga * (car_capacity / charger_power) * 60
        std = self.CAR_TYPES[car_type]["bateria_desviacion"] * (car_capacity / charger_power) * 60
        charging_time = max(1, np.random.normal(loc=mean, scale=std))

        # energía entregada en kWh
        energy_delivered = frac_carga * car_capacity

        yield self.env.timeout(charging_time)
        return charging_time, energy_delivered

# ------------------------------------------------------------------------------------
# 3. Funciones de llegada de coches y simulación con SimPy
# ------------------------------------------------------------------------------------

def get_inter_arrival_time(current_time, inter_arrival_df, MONTHS):
    """
    Devuelve el tiempo medio de llegada en minutos, según la fecha (mes y día de la semana).
    """
    month = MESES[current_time.month - 1]                # Mes en español
    weekday_name = DIAS_SEMANA[current_time.weekday()]   # Día en español
    return inter_arrival_df.loc[weekday_name, month]

def car(env, name, electrolinera, stats, precio_dia_mes_df,
        energy_by_car_type, energy_by_weekday, energy_by_month, start_datetime,
        served_by_car_type, abandoned_by_car_type):
    """
    Proceso de cada coche:
      - Selecciona el tipo de coche según probabilidades.
      - Calcula la fracción de batería a cargar.
      - Hace cola en un cargador y se va al terminar la carga.
    """
    arrival_time = env.now
    queue_length = sum(len(station["resource"].queue) for station in electrolinera.stations)
    electrolinera.queue_lengths.append(queue_length)

    car_type = random.choices(
        list(electrolinera.CAR_TYPES.keys()),
        weights=[electrolinera.CAR_TYPES[key]["probabilidad"] for key in electrolinera.CAR_TYPES]
    )[0]

    electrolinera.CAR_TYPES[car_type]["count"] += 1
    # Guardar el tipo en la variable stats
    stats["car_type"] = car_type

    media_bateria = electrolinera.CAR_TYPES[car_type]["bateria_media"]
    desv_bateria = electrolinera.CAR_TYPES[car_type]["bateria_desviacion"]
    frac_carga = np.random.normal(media_bateria, desv_bateria)
    frac_carga = np.clip(frac_carga, 0.01, 1.0)

    car_capacity = float(car_type[:-3])

    if car_capacity < 70:
        # Solo se consideran cargadores de 100kW o menos
        candidate_stations = [
            s for s in electrolinera.stations
            if s["power"] <= 100
        ]
    else:
        # Cualquier cargador vale
        candidate_stations = electrolinera.stations

    # Verificar colas en los cargadores candidatos
    available_stations = [
        s for s in candidate_stations
        if s["queue_limit"] is None or len(s["resource"].queue) < s["queue_limit"]
    ]

    if not available_stations:
        stats["abandoned"] += 1
        abandoned_by_car_type[car_type] += 1
        stats["system_times"].append(0)
        return



    station = min(available_stations, key=lambda s: len(s["resource"].queue))

    with station["resource"].request() as req:
        charger_type = station["type"]
        # 1) mido espera en cola
        wait_start = env.now
        yield req
        wait_time = env.now - wait_start
        electrolinera.stats_by_charger_type[charger_type]["queue_times"].append(wait_time)
        electrolinera.stats_by_charger_type[charger_type]["queue_lengths"].append(queue_length)


        # acumular totales
        electrolinera.charger_queue_times.append(wait_time)
        electrolinera.total_wait_time += wait_time
        if wait_time > 0:
            electrolinera.waiting_cars += 1

        # 2) arranca la carga
        charge_start = env.now
        charging_time, energy = yield env.process(
            electrolinera.charge(name, station, car_type, frac_carga)
        )

        electrolinera.stats_by_charger_type[charger_type]["charging_times"].append(charging_time)
        energy_by_car_type[car_type] += energy



        fecha = start_datetime + timedelta(minutes=env.now)
        mes_nombre = MESES[fecha.month - 1]              # "Enero", "Febrero", …
        dia_nombre = DIAS_SEMANA[fecha.weekday()]        # "Lunes", "Martes", …

        precio_compra = precio_dia_mes_df.loc[dia_nombre, mes_nombre]

        stats["coste_energia"] += energy * precio_compra
        energy_by_weekday[fecha.weekday()] += energy

        # si tus claves van de 1 a 12:
        energy_by_month[fecha.month] += energy

        charge_end = env.now

        # estadísticas de ocupación y energía
        station["busy_time"] += (charge_end - charge_start)
        electrolinera.stats_by_charger_type[charger_type]["busy_time"] += (charge_end - charge_start)
        electrolinera.stats_by_charger_type[charger_type]["served"] += 1
        electrolinera.charging_times.append(charging_time)
        electrolinera.energy_by_type[station["type"]] += energy

    # Al terminar la carga, damos el coche por servido y registramos su tiempo total
    stats["served"] += 1
    served_by_car_type[car_type] += 1
    stats["system_times"].append(charge_end - arrival_time)

def car_generator(
    env, electrolinera, stats,
    inter_arrival_df, MONTHS,
    SIM_TIME, precio_dia_mes_df,
    energy_by_car_type, energy_by_weekday, energy_by_month, start_datetime,
    served_by_car_type, abandoned_by_car_type
):

    """
    Genera coches de forma continua durante el tiempo de simulación.
    """
    i = 0
    current_time = datetime(2024, 1, 1)
    while True:
        iti = get_inter_arrival_time(current_time, inter_arrival_df, MONTHS)
        yield env.timeout(random.expovariate(1 / iti))

        # lanzamos un proceso 'car'
        env.process(
            car(
                env,
                f"Car {i}",
                electrolinera,
                stats,
                precio_dia_mes_df,
                energy_by_car_type,
                energy_by_weekday,
                energy_by_month,
                start_datetime,
                served_by_car_type,
                abandoned_by_car_type
            )
        )

        current_time += timedelta(minutes=iti)
        i += 1

def run_simulation_from_dfs(
    config_df, car_types_df, params_df,
    inter_arrival_df, econ_params_df, precio_dia_mes_df
):

    """
    Función principal que toma los DataFrames (o CSV subidos por el usuario),
    ejecuta la simulación y devuelve un DataFrame con los resultados.
    """
    CHARGERS = parse_configuration_df(config_df)
    CAR_TYPES = parse_car_types_df(car_types_df)
    SIM_TIME, NUM_REPLICATIONS, START_DATE = parse_simulation_params_df(params_df)
    inter_arrival_parsed, MONTHS = parse_inter_arrival_df(inter_arrival_df)

    def run_simulation(replication):
        resultados_dict = {}
        # Fijar semillas para reproducibilidad
        random.seed(42 + replication)
        np.random.seed(42 + replication)

        # — SUPUESTOS ECONÓMICOS —
        economics = dict(zip(econ_params_df["variable"], econ_params_df["valor"]))

        coste_mantenimiento_pct = float(economics["coste_mantenimiento_pct"])
        vida_util_cargador = float(economics["vida_util_cargador_anos"])
        coste_explotacion_anual = float(economics["coste_explotacion_anual_euros"])
        coste_fijo = float(economics["inversion_fija_por_cargador_euros"])
        coste_variable = float(economics["inversion_variable_cargador_euros/kW"])
        precio_potencia_anual_kW = float(economics.get("precio_potencia_anual_euros/kW", 0))


        # Potencia total instalada (suma de potencias de todos los cargadores)
        potencia_total_kW = sum(cfg["count"] * cfg["power"] for cfg in CHARGERS.values())

        # Coste anual de la potencia contratada
        coste_potencia = potencia_total_kW * precio_potencia_anual_kW

        # Fórmula de coste de cargador en función de su potencia
        def calcular_coste_cargador(potencia_kW):
            return coste_fijo + coste_variable * potencia_kW

        # Crear entorno y electrolinera
        env = simpy.Environment()
        electrolinera = Electrolinera(
            env,
            CHARGERS,
            CAR_TYPES
        )

        # Acumuladores detallados
        energy_by_car_type = {tipo: 0.0 for tipo in CAR_TYPES}
        served_by_car_type = {tipo: 0 for tipo in CAR_TYPES}
        abandoned_by_car_type = {tipo: 0 for tipo in CAR_TYPES}

        energy_by_weekday = {i: 0.0 for i in range(7)}       # 0=Lunes, 6=Domingo
        energy_by_month = {m: 0.0 for m in range(1, 13)}     # 1=Enero, ..., 12=Diciembre

        

        # Estadísticas básicas
        stats = {"served": 0, "abandoned": 0, "system_times": [], "coste_energia": 0.0}

# Arrancamos el generador de llegadas
        env.process(
        car_generator(
            env,
            electrolinera,
            stats,
            inter_arrival_parsed,
            MONTHS,
            SIM_TIME,
            precio_dia_mes_df,
            energy_by_car_type,
            energy_by_weekday,
            energy_by_month,
            START_DATE,
            served_by_car_type,
            abandoned_by_car_type
        )
    )



        # Ejecutar simulación
        env.run(until=SIM_TIME)

        # Conteo final por tipo de coche
        car_type_counts = {
            key: electrolinera.CAR_TYPES[key]["count"]
            for key in electrolinera.CAR_TYPES
        }

        # — CÁLCULO DE NUEVAS MÉTRICAS —
        sim_hours = SIM_TIME / 60.0
        num_chargers = sum(cfg["count"] for cfg in CHARGERS.values())
        days = SIM_TIME / (24 * 60)

        # 1) Espera
        mean_wait = (
            electrolinera.total_wait_time / stats["served"]
            if stats["served"] else 0
        )
        pct_wait = (
            electrolinera.waiting_cars / stats["served"] * 100
            if stats["served"] else 0
        )

        # 2) Carga
        mean_charge_time = (
            np.mean(electrolinera.charging_times)
            if electrolinera.charging_times else 0
        )

        # 3) Ocupación
        total_busy = sum(st["busy_time"] for st in electrolinera.stations)
        utilization = total_busy / (num_chargers * SIM_TIME) * 100

        # 4) Energía por cargador y hora
        total_energy = sum(electrolinera.energy_by_type.values())
        energy_per_charger_per_hour = total_energy / num_chargers / sim_hours

        # — ECONOMÍA SENCILLA —

        # 1) Ingresos por venta de energía
        ingresos = sum(
            electrolinera.energy_by_type[cargador] * CHARGERS[cargador]["precio_venta"]
            for cargador in CHARGERS
        )

        # 2) Coste de electricidad comprada
        coste_energia = stats["coste_energia"]
        start_datetime = datetime(2024, 1, 1)  # inicio de la simulación

        # 3) Coste de mantenimiento de todos los cargadores
        coste_mantenimiento = sum(
            cfg["count"] * calcular_coste_cargador(cfg["power"]) * coste_mantenimiento_pct
            for cfg in CHARGERS.values()
        )

        # 4) Coste de amortización de cargadores
        coste_amortizacion = sum(
            cfg["count"] * calcular_coste_cargador(cfg["power"]) / vida_util_cargador
            for cfg in CHARGERS.values()
        )

        # 5) Costes totales
        costes_totales = coste_energia + coste_mantenimiento + coste_amortizacion + coste_explotacion_anual + coste_potencia

        # 6) Beneficio neto
        beneficio_neto = ingresos - costes_totales

        # 7) Inversión inicial (CAPEX)
        inversion_inicial = sum(
            cfg["count"] * calcular_coste_cargador(cfg["power"])
            for cfg in CHARGERS.values()
        )

        # 8) Rentabilidad
        rentabilidad = (beneficio_neto / inversion_inicial) * 100 if inversion_inicial > 0 else 0

        # 5) Coches atendidos por día
        cars_per_day = stats["served"] / days

        dias_semana = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
        meses = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
                 "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]

        served_counts = {f"VE atendidos tipo {tipo}": count for tipo, count in served_by_car_type.items()}
        abandoned_counts = {f"VE que abandonan tipo {tipo}": count for tipo, count in abandoned_by_car_type.items()}


        # — DEVOLUCIÓN DE TODAS LAS MÉTRICAS —
        resultados_dict.update({
        "VE atendidos": stats["served"],
        "VE que abandonan sin servicio": stats["abandoned"],
        "Tasa de abandono (%)": (
            stats["abandoned"] / (stats["served"] + stats["abandoned"]) * 100
            if (stats["served"] + stats["abandoned"]) > 0 else 0
        ),
        "Tiempo promedio en el sistema (min)": (
            np.mean(stats["system_times"]) if stats["system_times"] else 0
        ),
        "Tiempo máximo en el sistema (min)": max(stats["system_times"], default=0),
        "Tiempo promedio en cola de cargadores (min)": (
            np.mean(electrolinera.charger_queue_times)
            if electrolinera.charger_queue_times else 0
        ),
        "Tiempo máximo en cola de cargadores (min)": max(electrolinera.charger_queue_times, default=0),
        "Longitud promedio de la cola": (
            np.mean(electrolinera.queue_lengths) if electrolinera.queue_lengths else 0
        ),
        "Longitud máxima de la cola": max(electrolinera.queue_lengths, default=0),
        **car_type_counts,
        **served_counts,
        **abandoned_counts,
        "% VE que esperan": pct_wait,
        "Tiempo medio de carga (min)": mean_charge_time,
        "Ocupación media cargadores (%)": utilization,
        "Energía/cargador/hora de media (kWh)": energy_per_charger_per_hour,
        "VE atendidos/día": cars_per_day,
        **{
            f"Energía total {ctype} (MWh)": energy / 1_000
            for ctype, energy in electrolinera.energy_by_type.items()
        },
        "Ingresos anuales (€)": ingresos,
        "Costes anuales (€)": costes_totales,
        "Beneficio neto anual (€)": beneficio_neto,
        "Rentabilidad sobre inversión (%)": rentabilidad,
        **{f"Energía total VE {tipo} (MWh)": energia / 1_000 for tipo, energia in energy_by_car_type.items()},
        **{f"Energía total {dias_semana[i]} (MWh)": energia / 1_000 for i, energia in energy_by_weekday.items()},
        **{f"Energía total {meses[m - 1]} (MWh)": energia / 1_000 for m, energia in energy_by_month.items()},

    })

        metricas_por_cargador = {}

        for charger_type, stats_ct in electrolinera.stats_by_charger_type.items():
            prefix = f"{charger_type} -"
            resultados_dict.update({
                f"{prefix} Tiempo máximo en cola (min)": max(stats_ct["queue_times"], default=0),
                f"{prefix} Tiempo medio de carga (min)": (
                    np.mean(stats_ct["charging_times"]) if stats_ct["charging_times"] else 0
                ),
                f"{prefix} Ocupación media (%)": (
                    stats_ct["busy_time"] / (CHARGERS[charger_type]["count"] * SIM_TIME) * 100
                    if SIM_TIME > 0 else 0
                ),
                f"{prefix} VE atendidos/día": (
                    stats_ct["served"] / days if days > 0 else 0
                )
            })

        resultados_dict.update(metricas_por_cargador)

        return resultados_dict



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
    config_file = st.file_uploader("Sube 'configuracion_cargadores.csv' ", type=["csv"])
    if config_file is not None:
        config_df = pd.read_csv(config_file)
        st.dataframe(config_df)
    else:
        st.info("Usando configuración de ejemplo.")
        config_df = pd.DataFrame({
            "tipo_cargador": ["100kW", "200kW"],
            "potencia_kW": [100, 200],
            "cantidad": [1, 1],
            "limite_cola": [1, 1],
            "precio_venta_euros/kWh": [0.35, 0.4]
        })
        st.dataframe(config_df)

    st.subheader("2) Tipos de VE")
    cars_file = st.file_uploader("Sube 'tipos_VE.csv'", type=["csv"])
    if cars_file is not None:
        car_types_df = pd.read_csv(cars_file)
        st.dataframe(car_types_df)
    else:
        st.info("Usando tipos de VE de ejemplo.")
        car_types_df = pd.DataFrame({
            "tipo_VE": ["120kWh", "70kWh", "40kWh"],
            "probabilidad": [0.2, 0.3, 0.5],
            "bateria_media": [0.8, 0.8, 0.8],
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
            "numero_replicaciones": [1],
            "fecha_inicio_simulacion": ["2030-01-01"]
        })
        st.dataframe(params_df)

    st.subheader("4) Tiempo entre la llegada de dos VE consecutivos en minutos (tiempo_entre_llegadas.csv)")
    inter_file = st.file_uploader("Sube 'tiempo_entre_llegadas.csv'", type=["csv"])

    if inter_file is not None:
        inter_arrival_df = pd.read_csv(inter_file, index_col=0)
        st.dataframe(inter_arrival_df)
    else:
        st.info("Usando tiempos de llegada de ejemplo.")
    inter_arrival_df = pd.DataFrame({
        'Enero':     [38.91253718, 38.89786902, 38.56352718, 40.57824997, 30.28524706, 47.08515923, 54.75541447],
        'Febrero':   [33.60159221, 37.44212252, 36.46307055, 32.14656999, 24.43452499, 41.0743448, 49.79206091],
        'Marzo':     [41.40633005, 39.35778498, 39.2871439, 36.10042619, 26.96984758, 43.25349866, 51.59155972],
        'Abril':     [33.02246629, 32.50386661, 27.74364064, 26.52906668, 25.82067659, 40.30428678, 40.85827409],
        'Mayo':      [38.01954985, 35.32538281, 35.10051157, 31.20770015, 22.48578719, 36.26718018, 45.90032584],
        'Junio':     [33.61410153, 34.91557558, 33.29799602, 29.67314742, 21.88623887, 34.51681692, 43.05498168],
        'Julio':     [31.34719558, 31.68545983, 31.42219539, 26.53393923, 20.60395424, 26.36734542, 36.25079934],
        'Agosto':    [27.34971897, 31.05207754, 31.53192892, 29.12047683, 22.18741331, 30.72059829, 39.51063492],
        'Septiembre':[36.59237222, 35.85493039, 35.42586698, 31.55396766, 23.45238402, 37.34919481, 44.84314798],
        'Octubre':   [35.93877445, 35.66374186, 35.21172993, 32.5874505, 23.21570443, 34.05305618, 33.8519743],
        'Noviembre': [37.15706933, 36.99152085, 36.94422196, 34.8196398, 26.16865658, 41.2267238, 46.02315563],
        'Diciembre': [35.92268886, 35.31674706, 33.71136546, 32.46588286, 24.25158394, 36.56828976, 41.22201836]
    }, index=["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"])


    st.dataframe(inter_arrival_df)


    st.subheader("5) Supuestos Económicos")
    econ_file = st.file_uploader("Sube 'parametros_economicos.csv'", type=["csv"])
    if econ_file is not None:
        econ_params_df = pd.read_csv(econ_file)
        st.dataframe(econ_params_df)
    else:
        st.info("Usando supuestos económicos de ejemplo.")
        econ_params_df = pd.DataFrame({
            "variable": [
                "coste_mantenimiento_pct",
                "vida_util_cargador_anos", "coste_explotacion_anual_euros",
                "inversion_fija_por_cargador_euros", "inversion_variable_cargador_euros/kW", "precio_potencia_anual_euros/kW"
            ],
            "valor": [
                0.07, 15, 50000,
                20000, 300, 50
            ]
        })
    st.dataframe(econ_params_df)

    st.subheader("6) Precio de compra por día y mes (€/kWh)")
    precio_dia_mes_file = st.file_uploader("Sube 'precio_compra_por_dia_y_mes.csv'", type=["csv"])
    if precio_dia_mes_file is not None:
        precio_dia_mes_df = pd.read_csv(precio_dia_mes_file, index_col=0)
        st.dataframe(precio_dia_mes_df)
    else:
        st.info("Usando precios de compra por día y mes de ejemplo.")
        precio_dia_mes_df = pd.DataFrame({
            "Enero":     [0.145, 0.144, 0.148, 0.152, 0.222, 0.189, 0.123],
            "Febrero":   [0.158, 0.161, 0.175, 0.198, 0.219, 0.185, 0.134],
            "Marzo":     [0.179, 0.179, 0.184, 0.202, 0.221, 0.162, 0.141],
            "Abril":     [0.201, 0.198, 0.195, 0.217, 0.214, 0.174, 0.162],
            "Mayo":      [0.223, 0.225, 0.224, 0.243, 0.236, 0.193, 0.179],
            "Junio":     [0.181, 0.176, 0.185, 0.200, 0.199, 0.167, 0.158],
            "Julio":     [0.164, 0.165, 0.167, 0.172, 0.176, 0.153, 0.144],
            "Agosto":    [0.142, 0.143, 0.145, 0.151, 0.157, 0.135, 0.124],
            "Septiembre":[0.132, 0.133, 0.134, 0.137, 0.140, 0.124, 0.113],
            "Octubre":   [0.153, 0.155, 0.156, 0.159, 0.161, 0.135, 0.122],
            "Noviembre": [0.172, 0.174, 0.176, 0.179, 0.181, 0.153, 0.131],
            "Diciembre": [0.193, 0.195, 0.197, 0.199, 0.202, 0.174, 0.147]
    }, index=["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"])

    st.dataframe(precio_dia_mes_df)

    if st.button("Ejecutar Simulación"):
        st.write("Ejecutando simulación, por favor espera...")
        results_df = run_simulation_from_dfs(config_df, car_types_df, params_df, inter_arrival_df, econ_params_df, precio_dia_mes_df)
        st.success("Simulación finalizada.")

        # 1) DataFrame completo
        st.subheader("Resultados por réplica")
        st.dataframe(results_df)

        # 2) Resumen de métricas (promedio sobre réplicas)
        st.subheader("Resumen global (media sobre réplicas)")
        summary_df = results_df.mean().to_frame(name="Promedio").round(2)
        st.table(summary_df)

        # 3) Botón de descarga de CSV con todos los datos
        csv_result = results_df.to_csv(index=False)
        st.download_button(
            label="Descargar resultados completos",
            data=csv_result,
            file_name="resultados_simulacion.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()



