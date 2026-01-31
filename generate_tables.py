#!/usr/bin/env python3
"""
Génère un rapport complet avec tableaux pour l'analyse des algorithmes de recherche
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from datetime import datetime

# Configuration du style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_results(filename='benchmark_results.csv'):
    """Charge les résultats depuis le fichier CSV"""
    df = pd.read_csv(filename)
    print(f"✓ Chargé {len(df)} résultats")
    return df

def generate_html_report(df, output_dir='graphs'):
    """Génère un rapport HTML complet avec tous les tableaux"""
    Path(output_dir).mkdir(exist_ok=True)

    html_content = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rapport d'Analyse - Algorithmes de Planification</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 10px;
        }}
        h3 {{
            color: #7f8c8d;
            margin-top: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th {{
            background-color: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }}
        td {{
            padding: 10px;
            border-bottom: 1px solid #ecf0f1;
        }}
        tr:hover {{
            background-color: #ecf0f1;
        }}
        .metric {{
            background-color: white;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #3498db;
        }}
        .metric-label {{
            color: #7f8c8d;
            font-size: 14px;
        }}
        .best {{
            background-color: #d4edda;
            font-weight: bold;
        }}
        .worst {{
            background-color: #f8d7da;
        }}
        .image-container {{
            text-align: center;
            margin: 20px 0;
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .info-box {{
            background-color: #e7f3ff;
            border-left: 4px solid #2196F3;
            padding: 15px;
            margin: 20px 0;
        }}
        .config-table {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .config-card {{
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        footer {{
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <h1>📊 Rapport d'Analyse des Algorithmes de Planification</h1>
    <p><strong>Date:</strong> {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>

    <div class="info-box">
        <h3>À propos de cette étude</h3>
        <p>Ce rapport présente une analyse comparative des algorithmes de recherche en utilisant
        le problème <strong>Dummy</strong> avec différentes configurations de taille (N) et de
        facteur de branchement (K).</p>
        <p><strong>Algorithmes testés:</strong> {', '.join(df['Algorithme'].unique())}</p>
        <p><strong>Nombre total de tests:</strong> {len(df)}</p>
        <p><strong>Configurations testées:</strong> {len(df.groupby(['N', 'K']))}</p>
    </div>
"""

    # 1. Tableau récapitulatif par algorithme
    html_content += generate_summary_table(df)

    # 2. Tableaux détaillés par configuration
    html_content += generate_config_tables(df)

    # 3. Tableau de comparaison directe
    html_content += generate_comparison_table(df)

    # 4. Analyse de la complexité
    html_content += generate_complexity_analysis(df)

    # 5. Tableaux de performances détaillées
    html_content += generate_detailed_performance_tables(df)

    # 6. Ajouter les graphiques
    html_content += add_graphics_section()

    # Footer
    html_content += """
    <footer>
        <p>Rapport généré automatiquement par analyze_results.py</p>
        <p>Projet: TreeSearchAndGames - Étude des algorithmes de planification</p>
    </footer>
</body>
</html>
"""

    # Sauvegarder le fichier HTML
    with open(f'{output_dir}/rapport_complet.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"✓ Rapport HTML sauvegardé: {output_dir}/rapport_complet.html")

def generate_summary_table(df):
    """Tableau récapitulatif par algorithme"""
    df_solved = df[df['Resolu'] == True].copy()

    stats = []
    for algo in sorted(df['Algorithme'].unique()):
        algo_data = df[df['Algorithme'] == algo]
        solved_data = df_solved[df_solved['Algorithme'] == algo]

        stats.append({
            'Algorithme': algo.upper(),
            'Tests Réussis': f"{len(solved_data)}/{len(algo_data)}",
            'Taux de Réussite (%)': f"{len(solved_data)/len(algo_data)*100:.1f}",
            'Temps Moyen (ms)': f"{solved_data['Temps_ms'].mean():.2f}" if len(solved_data) > 0 else 'N/A',
            'Temps Min (ms)': f"{solved_data['Temps_ms'].min():.2f}" if len(solved_data) > 0 else 'N/A',
            'Temps Max (ms)': f"{solved_data['Temps_ms'].max():.2f}" if len(solved_data) > 0 else 'N/A',
            'Nœuds Explorés (moy.)': f"{solved_data['Noeuds_Explores'].mean():.0f}" if len(solved_data) > 0 else 'N/A',
            'Longueur Solution (moy.)': f"{solved_data['Longueur_Solution'].mean():.1f}" if len(solved_data) > 0 else 'N/A',
        })

    stats_df = pd.DataFrame(stats)

    # Identifier les meilleurs et pires
    if len(df_solved) > 0:
        best_success = stats_df['Taux de Réussite (%)'].astype(float).idxmax()
        best_time = stats_df[stats_df['Temps Moyen (ms)'] != 'N/A']['Temps Moyen (ms)'].astype(float).idxmin()

    html = """
    <h2>1. Tableau Récapitulatif des Performances</h2>
    <table>
        <thead>
            <tr>
"""

    for col in stats_df.columns:
        html += f"                <th>{col}</th>\n"

    html += """
            </tr>
        </thead>
        <tbody>
"""

    for idx, row in stats_df.iterrows():
        row_class = ''
        if len(df_solved) > 0:
            if idx == best_success or (stats_df['Temps Moyen (ms)'].iloc[idx] != 'N/A' and idx == best_time):
                row_class = ' class="best"'

        html += f"            <tr{row_class}>\n"
        for col in stats_df.columns:
            html += f"                <td>{row[col]}</td>\n"
        html += "            </tr>\n"

    html += """
        </tbody>
    </table>
"""

    return html

def generate_config_tables(df):
    """Tableaux par configuration (N, K)"""
    html = """
    <h2>2. Performances par Configuration (N, K)</h2>
"""

    configs = df.groupby(['N', 'K']).size().reset_index()[['N', 'K']]

    for _, config in configs.iterrows():
        n, k = config['N'], config['K']
        config_data = df[(df['N'] == n) & (df['K'] == k)]

        html += f"""
    <h3>Configuration: N={n}, K={k} (Complexité: {n*k})</h3>
    <table>
        <thead>
            <tr>
                <th>Algorithme</th>
                <th>Résolu</th>
                <th>Temps Moyen (ms)</th>
                <th>Nœuds Explorés</th>
                <th>Longueur Solution</th>
            </tr>
        </thead>
        <tbody>
"""

        for algo in sorted(config_data['Algorithme'].unique()):
            algo_data = config_data[config_data['Algorithme'] == algo]
            solved = algo_data[algo_data['Resolu'] == True]

            resolu = f"{len(solved)}/{len(algo_data)}"
            temps = f"{solved['Temps_ms'].mean():.2f}" if len(solved) > 0 else 'N/A'
            noeuds = f"{solved['Noeuds_Explores'].mean():.0f}" if len(solved) > 0 else 'N/A'
            longueur = f"{solved['Longueur_Solution'].mean():.1f}" if len(solved) > 0 else 'N/A'

            html += f"""
            <tr>
                <td><strong>{algo.upper()}</strong></td>
                <td>{resolu}</td>
                <td>{temps}</td>
                <td>{noeuds}</td>
                <td>{longueur}</td>
            </tr>
"""

        html += """
        </tbody>
    </table>
"""

    return html

def generate_comparison_table(df):
    """Tableau de comparaison directe entre algorithmes"""
    df_solved = df[df['Resolu'] == True].copy()

    html = """
    <h2>3. Tableau de Comparaison Directe</h2>
    <p>Pour chaque métrique, le meilleur algorithme est mis en évidence.</p>
"""

    metrics = {
        'Taux de Réussite (%)': ('Resolu', lambda x: x.sum() / len(x) * 100, True),
        'Temps Moyen (ms)': ('Temps_ms', 'mean', False),
        'Temps Médian (ms)': ('Temps_ms', 'median', False),
        'Nœuds Explorés': ('Noeuds_Explores', 'mean', False),
        'Longueur Solution': ('Longueur_Solution', 'mean', False),
        'Efficacité (ms/nœud)': ('Temps_ms', lambda x: (df_solved.loc[x.index, 'Temps_ms'] / df_solved.loc[x.index, 'Noeuds_Explores']).mean(), False),
    }

    comparison_data = []
    for algo in sorted(df['Algorithme'].unique()):
        algo_full = df[df['Algorithme'] == algo]
        algo_solved = df_solved[df_solved['Algorithme'] == algo]

        row = {'Algorithme': algo.upper()}

        for metric_name, (col, agg, use_full) in metrics.items():
            data = algo_full if use_full else algo_solved
            if len(data) > 0:
                if callable(agg):
                    value = agg(data[col] if col in data.columns else data)
                else:
                    value = data[col].agg(agg)
                row[metric_name] = value
            else:
                row[metric_name] = np.nan

        comparison_data.append(row)

    comp_df = pd.DataFrame(comparison_data)

    html += """
    <table>
        <thead>
            <tr>
                <th>Algorithme</th>
"""

    for metric in list(metrics.keys()):
        html += f"                <th>{metric}</th>\n"

    html += """
            </tr>
        </thead>
        <tbody>
"""

    # Identifier les meilleurs pour chaque métrique
    best_indices = {}
    for metric, (_, _, higher_better) in metrics.items():
        if higher_better:
            best_indices[metric] = comp_df[metric].idxmax()
        else:
            best_indices[metric] = comp_df[metric].idxmin()

    for idx, row in comp_df.iterrows():
        html += "            <tr>\n"
        html += f"                <td><strong>{row['Algorithme']}</strong></td>\n"

        for metric in list(metrics.keys()):
            value = row[metric]
            cell_class = ' class="best"' if idx == best_indices.get(metric) else ''
            value_str = f"{value:.2f}" if not pd.isna(value) else 'N/A'
            html += f"                <td{cell_class}>{value_str}</td>\n"

        html += "            </tr>\n"

    html += """
        </tbody>
    </table>
"""

    return html

def generate_complexity_analysis(df):
    """Analyse de la complexité"""
    df_solved = df[df['Resolu'] == True].copy()
    df_solved['Complexite'] = df_solved['N'] * df_solved['K']

    html = """
    <h2>4. Analyse de la Complexité</h2>
    <p>Impact de la complexité (N × K) sur les performances.</p>

    <table>
        <thead>
            <tr>
                <th>Plage de Complexité</th>
                <th>Algorithme</th>
                <th>Nb Tests Réussis</th>
                <th>Temps Moyen (ms)</th>
                <th>Nœuds Moyens</th>
            </tr>
        </thead>
        <tbody>
"""

    # Définir des plages de complexité
    ranges = [
        (0, 500, "Très Simple (< 500)"),
        (500, 2000, "Simple (500-2000)"),
        (2000, 10000, "Moyen (2000-10000)"),
        (10000, 100000, "Difficile (> 10000)")
    ]

    for min_val, max_val, label in ranges:
        range_data = df_solved[(df_solved['Complexite'] >= min_val) & (df_solved['Complexite'] < max_val)]

        if len(range_data) == 0:
            continue

        for algo in sorted(range_data['Algorithme'].unique()):
            algo_data = range_data[range_data['Algorithme'] == algo]

            html += f"""
            <tr>
                <td>{label}</td>
                <td><strong>{algo.upper()}</strong></td>
                <td>{len(algo_data)}</td>
                <td>{algo_data['Temps_ms'].mean():.2f}</td>
                <td>{algo_data['Noeuds_Explores'].mean():.0f}</td>
            </tr>
"""

    html += """
        </tbody>
    </table>
"""

    return html

def generate_detailed_performance_tables(df):
    """Tableaux de performances détaillées"""
    df_solved = df[df['Resolu'] == True].copy()

    html = """
    <h2>5. Performances Détaillées</h2>
"""

    # 5.1 Top 10 des meilleures performances
    html += """
    <h3>5.1 Top 10 des Meilleures Performances (Temps)</h3>
    <table>
        <thead>
            <tr>
                <th>Rang</th>
                <th>Algorithme</th>
                <th>N</th>
                <th>K</th>
                <th>Temps (ms)</th>
                <th>Nœuds</th>
                <th>Longueur</th>
            </tr>
        </thead>
        <tbody>
"""

    top10 = df_solved.nsmallest(10, 'Temps_ms')
    for idx, (_, row) in enumerate(top10.iterrows(), 1):
        html += f"""
            <tr>
                <td>{idx}</td>
                <td><strong>{row['Algorithme'].upper()}</strong></td>
                <td>{row['N']}</td>
                <td>{row['K']}</td>
                <td>{row['Temps_ms']:.2f}</td>
                <td>{row['Noeuds_Explores']:.0f}</td>
                <td>{row['Longueur_Solution']:.0f}</td>
            </tr>
"""

    html += """
        </tbody>
    </table>
"""

    # 5.2 Tableau des écarts-types
    html += """
    <h3>5.2 Stabilité des Algorithmes (Écart-type)</h3>
    <p>Plus l'écart-type est faible, plus l'algorithme est stable.</p>
    <table>
        <thead>
            <tr>
                <th>Algorithme</th>
                <th>Écart-Type Temps (ms)</th>
                <th>Écart-Type Nœuds</th>
                <th>Coefficient de Variation (%)</th>
            </tr>
        </thead>
        <tbody>
"""

    for algo in sorted(df_solved['Algorithme'].unique()):
        algo_data = df_solved[df_solved['Algorithme'] == algo]
        std_time = algo_data['Temps_ms'].std()
        mean_time = algo_data['Temps_ms'].mean()
        cv = (std_time / mean_time * 100) if mean_time > 0 else 0

        html += f"""
            <tr>
                <td><strong>{algo.upper()}</strong></td>
                <td>{std_time:.2f}</td>
                <td>{algo_data['Noeuds_Explores'].std():.2f}</td>
                <td>{cv:.2f}%</td>
            </tr>
"""

    html += """
        </tbody>
    </table>
"""

    return html

def add_graphics_section():
    """Ajoute la section avec les graphiques"""
    html = """
    <h2>6. Visualisations Graphiques</h2>
"""

    graphs = [
        ('temps_execution.png', 'Temps d\'Exécution'),
        ('noeuds_explores.png', 'Nœuds Explorés'),
        ('taux_reussite.png', 'Taux de Réussite'),
        ('qualite_solution.png', 'Qualité de la Solution'),
        ('comparaison_complexite.png', 'Comparaison selon la Complexité')
    ]

    for filename, title in graphs:
        if Path(f'graphs/{filename}').exists():
            html += f"""
    <div class="image-container">
        <h3>{title}</h3>
        <img src="{filename}" alt="{title}">
    </div>
"""

    return html

def generate_excel_tables(df, output_dir='graphs'):
    """Génère des tableaux Excel détaillés"""
    Path(output_dir).mkdir(exist_ok=True)

    with pd.ExcelWriter(f'{output_dir}/tableaux_detailles.xlsx', engine='openpyxl') as writer:
        # Feuille 1: Données brutes
        df.to_excel(writer, sheet_name='Données Brutes', index=False)

        # Feuille 2: Résumé par algorithme
        df_solved = df[df['Resolu'] == True]
        summary = df.groupby('Algorithme').agg({
            'Resolu': ['sum', 'count', lambda x: x.sum()/len(x)*100],
            'Temps_ms': ['mean', 'std', 'min', 'max', 'median'],
            'Noeuds_Explores': ['mean', 'std', 'min', 'max'],
            'Longueur_Solution': ['mean', 'std', 'min', 'max']
        }).round(2)
        summary.to_excel(writer, sheet_name='Résumé par Algorithme')

        # Feuille 3: Par configuration
        config_summary = df.groupby(['N', 'K', 'Algorithme']).agg({
            'Resolu': 'mean',
            'Temps_ms': 'mean',
            'Noeuds_Explores': 'mean',
            'Longueur_Solution': 'mean'
        }).round(2)
        config_summary.to_excel(writer, sheet_name='Par Configuration')

        # Feuille 4: Comparaison directe
        pivot_time = df_solved.pivot_table(values='Temps_ms', index='Algorithme', columns=['N', 'K'], aggfunc='mean').round(2)
        pivot_time.to_excel(writer, sheet_name='Temps par Config')

        # Feuille 5: Taux de réussite
        pivot_success = df.pivot_table(values='Resolu', index='Algorithme', columns=['N', 'K'], aggfunc='mean').round(3) * 100
        pivot_success.to_excel(writer, sheet_name='Taux Réussite %')

    print(f"✓ Tableaux Excel sauvegardés: {output_dir}/tableaux_detailles.xlsx")

def main():
    """Fonction principale"""
    print("="*80)
    print("GÉNÉRATION DES TABLEAUX ET DU RAPPORT")
    print("="*80 + "\n")

    # Charger les données
    try:
        df = load_results('benchmark_results.csv')
    except FileNotFoundError:
        print("❌ Fichier 'benchmark_results.csv' non trouvé!")
        return

    print(f"\nNombre total de tests: {len(df)}")
    print(f"Algorithmes testés: {', '.join(df['Algorithme'].unique())}\n")

    # Générer le rapport HTML
    print("Génération du rapport HTML...")
    generate_html_report(df)

    # Générer les tableaux Excel
    print("\nGénération des tableaux Excel...")
    try:
        generate_excel_tables(df)
    except Exception as e:
        print(f"⚠ Erreur lors de la génération Excel: {e}")
        print("  (Le package openpyxl est peut-être manquant: pip install openpyxl)")

    print("\n" + "="*80)
    print("✓ GÉNÉRATION TERMINÉE")
    print("="*80)
    print("\nFichiers générés:")
    print("  - graphs/rapport_complet.html (ouvrir dans un navigateur)")
    print("  - graphs/tableaux_detailles.xlsx (ouvrir avec Excel)")
    print()

if __name__ == '__main__':
    main()
