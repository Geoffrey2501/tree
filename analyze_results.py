#!/usr/bin/env python3
"""
Script d'analyse et visualisation des résultats de benchmark
Génère des graphiques comparatifs pour les algorithmes de recherche
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Configuration du style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_results(filename='benchmark_results.csv'):
    """Charge les résultats depuis le fichier CSV"""
    df = pd.read_csv(filename)
    print(f"✓ Chargé {len(df)} résultats")
    return df

def plot_temps_execution(df, output_dir='graphs'):
    """Graphique: Temps d'exécution par algorithme et taille de problème"""
    Path(output_dir).mkdir(exist_ok=True)

    # Filtrer uniquement les problèmes résolus
    df_solved = df[df['Resolu'] == True].copy()

    # Créer une colonne pour identifier la configuration
    df_solved['Config'] = df_solved['N'].astype(str) + '_k' + df_solved['K'].astype(str)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Temps moyen par algorithme et N
    ax1 = axes[0, 0]
    pivot_time = df_solved.groupby(['Algorithme', 'N'])['Temps_ms'].mean().reset_index()
    for algo in df_solved['Algorithme'].unique():
        data = pivot_time[pivot_time['Algorithme'] == algo]
        ax1.plot(data['N'], data['Temps_ms'], marker='o', label=algo.upper(), linewidth=2)
    ax1.set_xlabel('Taille du problème (N)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Temps moyen (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Temps d\'exécution en fonction de N', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)

    # 2. Temps moyen par algorithme et K
    ax2 = axes[0, 1]
    pivot_k = df_solved.groupby(['Algorithme', 'K'])['Temps_ms'].mean().reset_index()
    for algo in df_solved['Algorithme'].unique():
        data = pivot_k[pivot_k['Algorithme'] == algo]
        ax2.plot(data['K'], data['Temps_ms'], marker='s', label=algo.upper(), linewidth=2)
    ax2.set_xlabel('Facteur de branchement (K)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Temps moyen (ms)', fontsize=12, fontweight='bold')
    ax2.set_title('Temps d\'exécution en fonction de K', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    # 3. Boxplot par algorithme
    ax3 = axes[1, 0]
    df_solved.boxplot(column='Temps_ms', by='Algorithme', ax=ax3)
    ax3.set_xlabel('Algorithme', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Temps (ms)', fontsize=12, fontweight='bold')
    ax3.set_title('Distribution des temps d\'exécution', fontsize=14, fontweight='bold')
    ax3.set_yscale('log')
    plt.sca(ax3)
    plt.xticks(rotation=45)

    # 4. Heatmap temps moyen
    ax4 = axes[1, 1]
    pivot_heat = df_solved.pivot_table(values='Temps_ms', index='Algorithme',
                                        columns='N', aggfunc='mean')
    sns.heatmap(pivot_heat, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax4,
                cbar_kws={'label': 'Temps (ms)'})
    ax4.set_title('Temps moyen par algorithme et N', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Taille N', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Algorithme', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/temps_execution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Graphique sauvegardé: {output_dir}/temps_execution.png")
    plt.close()

def plot_noeuds_explores(df, output_dir='graphs'):
    """Graphique: Nombre de nœuds explorés"""
    Path(output_dir).mkdir(exist_ok=True)

    df_solved = df[df['Resolu'] == True].copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Nœuds explorés vs N
    ax1 = axes[0]
    pivot_nodes = df_solved.groupby(['Algorithme', 'N'])['Noeuds_Explores'].mean().reset_index()
    for algo in df_solved['Algorithme'].unique():
        data = pivot_nodes[pivot_nodes['Algorithme'] == algo]
        ax1.plot(data['N'], data['Noeuds_Explores'], marker='o', label=algo.upper(), linewidth=2)
    ax1.set_xlabel('Taille du problème (N)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Nœuds explorés (moyenne)', fontsize=12, fontweight='bold')
    ax1.set_title('Nombre de nœuds explorés en fonction de N', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)

    # 2. Efficacité: Temps par nœud
    ax2 = axes[1]
    df_solved['Temps_par_noeud'] = df_solved['Temps_ms'] / df_solved['Noeuds_Explores']
    pivot_eff = df_solved.groupby(['Algorithme', 'N'])['Temps_par_noeud'].mean().reset_index()
    for algo in df_solved['Algorithme'].unique():
        data = pivot_eff[pivot_eff['Algorithme'] == algo]
        ax2.plot(data['N'], data['Temps_par_noeud'], marker='s', label=algo.upper(), linewidth=2)
    ax2.set_xlabel('Taille du problème (N)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Temps par nœud (ms)', fontsize=12, fontweight='bold')
    ax2.set_title('Efficacité: Temps de traitement par nœud', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/noeuds_explores.png', dpi=300, bbox_inches='tight')
    print(f"✓ Graphique sauvegardé: {output_dir}/noeuds_explores.png")
    plt.close()

def plot_taux_reussite(df, output_dir='graphs'):
    """Graphique: Taux de réussite par algorithme"""
    Path(output_dir).mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Taux de réussite global
    ax1 = axes[0]
    success_rate = df.groupby('Algorithme')['Resolu'].agg(['sum', 'count'])
    success_rate['taux'] = (success_rate['sum'] / success_rate['count'] * 100)
    success_rate = success_rate.sort_values('taux', ascending=False)

    bars = ax1.bar(range(len(success_rate)), success_rate['taux'],
                   color=plt.cm.viridis(np.linspace(0, 1, len(success_rate))))
    ax1.set_xticks(range(len(success_rate)))
    ax1.set_xticklabels(success_rate.index.str.upper(), rotation=45)
    ax1.set_ylabel('Taux de réussite (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Taux de réussite global par algorithme', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3, axis='y')

    # Ajouter les valeurs sur les barres
    for i, (bar, val) in enumerate(zip(bars, success_rate['taux'])):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

    # 2. Taux de réussite par taille de problème
    ax2 = axes[1]
    success_by_n = df.groupby(['N', 'Algorithme'])['Resolu'].mean().reset_index()
    success_by_n['taux'] = success_by_n['Resolu'] * 100

    for algo in df['Algorithme'].unique():
        data = success_by_n[success_by_n['Algorithme'] == algo]
        ax2.plot(data['N'], data['taux'], marker='o', label=algo.upper(), linewidth=2)

    ax2.set_xlabel('Taille du problème (N)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Taux de réussite (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Taux de réussite en fonction de N', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.set_xscale('log')
    ax2.set_ylim(-5, 105)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/taux_reussite.png', dpi=300, bbox_inches='tight')
    print(f"✓ Graphique sauvegardé: {output_dir}/taux_reussite.png")
    plt.close()

def plot_qualite_solution(df, output_dir='graphs'):
    """Graphique: Qualité de la solution (longueur du chemin)"""
    Path(output_dir).mkdir(exist_ok=True)

    df_solved = df[df['Resolu'] == True].copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Longueur moyenne de solution
    ax1 = axes[0]
    pivot_len = df_solved.groupby(['Algorithme', 'N'])['Longueur_Solution'].mean().reset_index()
    for algo in df_solved['Algorithme'].unique():
        data = pivot_len[pivot_len['Algorithme'] == algo]
        ax1.plot(data['N'], data['Longueur_Solution'], marker='o', label=algo.upper(), linewidth=2)
    ax1.set_xlabel('Taille du problème (N)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Longueur moyenne de la solution', fontsize=12, fontweight='bold')
    ax1.set_title('Qualité de la solution: Longueur du chemin', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)

    # 2. Comparaison: Longueur de solution par algorithme
    ax2 = axes[1]
    df_solved.boxplot(column='Longueur_Solution', by='Algorithme', ax=ax2)
    ax2.set_xlabel('Algorithme', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Longueur de la solution', fontsize=12, fontweight='bold')
    ax2.set_title('Distribution des longueurs de solution', fontsize=14, fontweight='bold')
    plt.sca(ax2)
    plt.xticks(rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/qualite_solution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Graphique sauvegardé: {output_dir}/qualite_solution.png")
    plt.close()

def plot_comparaison_complexite(df, output_dir='graphs'):
    """Graphique: Comparaison selon la complexité (N × K)"""
    Path(output_dir).mkdir(exist_ok=True)

    df_solved = df[df['Resolu'] == True].copy()
    df_solved['Complexite'] = df_solved['N'] * df_solved['K']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Temps vs Complexité
    ax1 = axes[0, 0]
    for algo in df_solved['Algorithme'].unique():
        data = df_solved[df_solved['Algorithme'] == algo]
        ax1.scatter(data['Complexite'], data['Temps_ms'], label=algo.upper(),
                   alpha=0.6, s=50)
    ax1.set_xlabel('Complexité (N × K)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Temps (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Temps d\'exécution vs Complexité', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    # 2. Nœuds vs Complexité
    ax2 = axes[0, 1]
    for algo in df_solved['Algorithme'].unique():
        data = df_solved[df_solved['Algorithme'] == algo]
        ax2.scatter(data['Complexite'], data['Noeuds_Explores'], label=algo.upper(),
                   alpha=0.6, s=50)
    ax2.set_xlabel('Complexité (N × K)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Nœuds explorés', fontsize=12, fontweight='bold')
    ax2.set_title('Nœuds explorés vs Complexité', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    # 3. Impact de K sur le temps (pour différents N)
    ax3 = axes[1, 0]
    n_values = sorted(df_solved['N'].unique())[:4]  # Prendre les 4 premiers N
    for n in n_values:
        data = df_solved[df_solved['N'] == n].groupby(['K', 'Algorithme'])['Temps_ms'].mean().reset_index()
        # Moyenne sur tous les algos pour cette vue
        avg_data = data.groupby('K')['Temps_ms'].mean().reset_index()
        ax3.plot(avg_data['K'], avg_data['Temps_ms'], marker='o', label=f'N={n}', linewidth=2)
    ax3.set_xlabel('Facteur de branchement (K)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Temps moyen (ms)', fontsize=12, fontweight='bold')
    ax3.set_title('Impact de K pour différentes valeurs de N', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)

    # 4. Comparaison performance relative
    ax4 = axes[1, 1]
    # Normaliser par rapport au meilleur algorithme pour chaque config
    perf_data = []
    for (n, k), group in df_solved.groupby(['N', 'K']):
        min_time = group['Temps_ms'].min()
        for _, row in group.iterrows():
            perf_data.append({
                'Algorithme': row['Algorithme'],
                'N': n,
                'K': k,
                'Performance_Relative': row['Temps_ms'] / min_time if min_time > 0 else 1
            })

    df_perf = pd.DataFrame(perf_data)
    avg_perf = df_perf.groupby('Algorithme')['Performance_Relative'].mean().sort_values()

    bars = ax4.barh(range(len(avg_perf)), avg_perf.values,
                    color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(avg_perf))))
    ax4.set_yticks(range(len(avg_perf)))
    ax4.set_yticklabels(avg_perf.index.str.upper())
    ax4.set_xlabel('Performance relative (1 = meilleur)', fontsize=12, fontweight='bold')
    ax4.set_title('Performance relative moyenne', fontsize=14, fontweight='bold')
    ax4.axvline(x=1, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Optimal')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='x')

    # Ajouter les valeurs
    for i, (bar, val) in enumerate(zip(bars, avg_perf.values)):
        ax4.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}x', va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparaison_complexite.png', dpi=300, bbox_inches='tight')
    print(f"✓ Graphique sauvegardé: {output_dir}/comparaison_complexite.png")
    plt.close()

def generate_statistics_table(df, output_dir='graphs'):
    """Génère des tableaux statistiques"""
    Path(output_dir).mkdir(exist_ok=True)

    df_solved = df[df['Resolu'] == True].copy()

    # Statistiques par algorithme
    stats = df.groupby('Algorithme').agg({
        'Resolu': ['sum', 'count', lambda x: (x.sum() / len(x) * 100)],
        'Temps_ms': ['mean', 'std', 'min', 'max'],
        'Noeuds_Explores': ['mean', 'std'],
        'Longueur_Solution': ['mean', 'std']
    }).round(2)

    stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
    stats = stats.rename(columns={
        'Resolu_sum': 'Nb_Reussis',
        'Resolu_count': 'Total_Tests',
        'Resolu_<lambda>': 'Taux_Reussite_%',
        'Temps_ms_mean': 'Temps_Moyen_ms',
        'Temps_ms_std': 'Temps_Ecart_Type_ms',
        'Temps_ms_min': 'Temps_Min_ms',
        'Temps_ms_max': 'Temps_Max_ms',
        'Noeuds_Explores_mean': 'Noeuds_Moyen',
        'Noeuds_Explores_std': 'Noeuds_Ecart_Type',
        'Longueur_Solution_mean': 'Longueur_Moyenne',
        'Longueur_Solution_std': 'Longueur_Ecart_Type'
    })

    stats.to_csv(f'{output_dir}/statistiques_par_algorithme.csv')
    print(f"✓ Tableau sauvegardé: {output_dir}/statistiques_par_algorithme.csv")

    # Afficher dans la console
    print("\n" + "="*80)
    print("STATISTIQUES PAR ALGORITHME")
    print("="*80)
    print(stats.to_string())
    print("="*80 + "\n")

    # Statistiques par configuration
    config_stats = df.groupby(['N', 'K']).agg({
        'Resolu': lambda x: (x.sum() / len(x) * 100),
        'Temps_ms': 'mean',
        'Noeuds_Explores': 'mean'
    }).round(2)

    config_stats = config_stats.rename(columns={
        'Resolu': 'Taux_Reussite_%',
        'Temps_ms': 'Temps_Moyen_ms',
        'Noeuds_Explores': 'Noeuds_Moyen'
    })

    config_stats.to_csv(f'{output_dir}/statistiques_par_configuration.csv')
    print(f"✓ Tableau sauvegardé: {output_dir}/statistiques_par_configuration.csv")

def main():
    """Fonction principale"""
    print("="*80)
    print("ANALYSE DES RÉSULTATS DE BENCHMARK")
    print("="*80 + "\n")

    # Charger les données
    try:
        df = load_results('benchmark_results.csv')
    except FileNotFoundError:
        print("❌ Fichier 'benchmark_results.csv' non trouvé!")
        print("   Veuillez d'abord exécuter BenchmarkRecherche.java")
        return

    print(f"\nNombre total de tests: {len(df)}")
    print(f"Algorithmes testés: {', '.join(df['Algorithme'].unique())}")
    print(f"Configurations testées: {len(df.groupby(['N', 'K']))}")
    print()

    # Générer les graphiques
    print("Génération des graphiques...\n")
    plot_temps_execution(df)
    plot_noeuds_explores(df)
    plot_taux_reussite(df)
    plot_qualite_solution(df)
    plot_comparaison_complexite(df)

    # Générer les statistiques
    print("\nGénération des statistiques...\n")
    generate_statistics_table(df)

    print("\n" + "="*80)
    print("✓ ANALYSE TERMINÉE")
    print("="*80)
    print("\nFichiers générés dans le dossier 'graphs/':")
    print("  - temps_execution.png")
    print("  - noeuds_explores.png")
    print("  - taux_reussite.png")
    print("  - qualite_solution.png")
    print("  - comparaison_complexite.png")
    print("  - statistiques_par_algorithme.csv")
    print("  - statistiques_par_configuration.csv")
    print()

if __name__ == '__main__':
    main()
