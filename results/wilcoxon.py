#!/usr/bin/env python3
import pandas as pd
from scipy import stats

def main():
    # 1. Cargar los datos crudos
    df = pd.read_csv('resultados_benchmark_master_OK.csv')

    # 2. Emparejamiento (Pairing)
    # Como los ensayos se corrieron secuencialmente, les asignamos un ID (del 1 al 50)
    # ordenándolos por el Timestamp para cada algoritmo.
    df['Trial_ID'] = df.groupby('Algorithm')['Timestamp'].rank(method='first').astype(int)

    # 3. Consolidar métricas a nivel bimanual (Ensayo completo)
    # Sumamos el esfuerzo de los brazos para JPL y CE, y promediamos el RMSE
    trial_df = df.groupby(['Algorithm', 'Trial_ID']).agg({
        'RMSE_m': 'mean',      
        'TJV_rad': 'sum',      # Este es nuestro Joint Path Length (JPL)
        'Jerk_rad_s': 'sum'    # Este es nuestro Control Effort (CE)
    }).reset_index()

    # 4. Pivotar la tabla para poner a los algoritmos frente a frente
    # Esto crea una tabla donde cada fila es un ensayo (1-50) y las columnas comparan a LEI vs LUMELSKY
    paired_df = trial_df.pivot(index='Trial_ID', columns='Algorithm', values=['RMSE_m', 'TJV_rad', 'Jerk_rad_s'])

    # 5. Función auxiliar para correr el test e imprimir bonito
    def correr_wilcoxon(metrica_csv, nombre_publicacion):
        data_lei = paired_df[(metrica_csv, 'LEI')]
        data_propuesto = paired_df[(metrica_csv, 'LUMELSKY')]
        
        # ¡LA MAGIA OCURRE AQUÍ!
        # Le pasamos las dos listas pareadas de 50 elementos a Scipy
        stat, p_val = stats.wilcoxon(data_lei, data_propuesto)
        
        print(f"--- Análisis para: {nombre_publicacion} ---")
        print(f"Media Baseline (Lei): {data_lei.mean():.4f}")
        print(f"Media Propuesto:      {data_propuesto.mean():.4f}")
        print(f"Estadístico W:        {stat}")
        
        # Imprimimos el p-value en notación científica para ver los ceros
        print(f"Valor p (p-value):    {p_val:.2e}")
        
        if p_val < 0.05:
            print("Resultado: ¡La diferencia es ESTADÍSTICAMENTE SIGNIFICATIVA! (p < 0.05)\n")
        else:
            print("Resultado: La diferencia NO es significativa.\n")

    # 6. Ejecutar el análisis
    print("======================================================")
    print(" RESULTADOS DEL WILCOXON SIGNED-RANK TEST (N=50 pares)")
    print("======================================================\n")

    correr_wilcoxon('RMSE_m', 'Tracking Accuracy (RMSE)')
    correr_wilcoxon('Jerk_rad_s', 'Control Effort (CE)')
    correr_wilcoxon('TJV_rad', 'Joint Path Length (JPL)')

if __name__ == '__main__':
    main()