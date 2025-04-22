import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_risks(data_path, data_sa_path):
    data = pd.read_excel(data_path)
    data_SA = pd.read_excel(data_sa_path)
    values_conc = []
    for metabolite in data['Маркер / Соотношение']:
        values_conc.append(float(data_SA.loc[0, metabolite]))
    data['Sample'] = values_conc
    risks = []
    for index, row in data.iterrows():
        if row['norm_1'] < row['Sample'] < row['norm_2']:
            risks.append(0)
        elif row['High_risk_1'] < row['Sample'] < row['norm_1'] or row['norm_2'] < row['Sample'] < row['High_risk_2']:
            risks.append(1)
        else:
            risks.append(2)
    data['Risks'] = risks
    data['Corrected_risks'] = data['Risks'] * data['веса']
    data['Weights_for_formula'] = data['веса'] * 2
    Risk_groups = data['Группа_риска'].unique()
    RISK_values = []
    for risk_group in Risk_groups:
        group_data = data[data['Группа_риска'] == risk_group]
        sum_corrected = group_data['Corrected_risks'].sum()
        sum_weights = group_data['Weights_for_formula'].sum()
        risk_score = (sum_corrected / sum_weights) * 10
        RISK_values.append(10-risk_score)
    df_risks = pd.DataFrame({'Группа риска': Risk_groups, 'Риск-скор': RISK_values})
    df_risks['Риск-скор'] = df_risks['Риск-скор'].round(0)
    return df_risks

st.title('Расчет риска по метаболитам')

# Пути к файлам
calculation_file = st.file_uploader('Загрузите расчетный файл', type=['xlsx'])
patient_file = st.file_uploader('Загрузите файл пациента', type=['xlsx'])

if calculation_file and patient_file:
    # Сохраняем файлы во временные переменные
    df_result = calculate_risks(calculation_file, patient_file)
    st.write('Результаты:')
    st.dataframe(df_result)

     # --- Построение радарной диаграммы ---
    labels = df_result['Группа риска'].tolist()
    risk_levels = df_result['Риск-скор'].tolist()

    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Замкнуть круг
    risk_levels += risk_levels[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Настройка градиентного фона
# Закрашивание пространства между окружностями
    colors = [
        'darkred',  # 0-1
        'red',      # 1-2
        'red',      # 2-3
        'darkorange', # 3-4
        'orange',   # 4-5
        'gold',     # 5-6
        'yellow',   # 6-7
        'yellowgreen', # 7-8
        'limegreen', # 8-9
        'green',    # 9-10
        'green'     # 10-11
    ]

    # Уровни для закрашивания
    levels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Закрашивание каждого уровня
    for i in range(len(levels) - 1):
        ax.fill_between(angles, levels[i], levels[i + 1], color=colors[i], alpha=0.3)
 
    # Построение
    ax.fill(angles, risk_levels, color='blue', alpha=0.25)
    ax.plot(angles, risk_levels, color='blue', linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.tick_params(axis='x', pad=-5)  # увеличьте pad для отступа


    ax.set_ylim(0, 10)
    ax.set_title('Радиальная диаграмма уровней риска', size=15)

    # Отображение графика в Streamlit
    st.pyplot(fig)
