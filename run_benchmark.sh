#!/bin/bash
# Script pour compiler, exécuter le benchmark et générer les graphiques

set -e  # Arrêter en cas d'erreur

echo "======================================================================"
echo "ÉTUDE COMPARATIVE DES ALGORITHMES DE PLANIFICATION"
echo "======================================================================"
echo ""

# Couleurs pour l'affichage
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Répertoire de travail
cd "$(dirname "$0")"

# Étape 1: Compilation
echo -e "${BLUE}[1/3] Compilation du projet...${NC}"
echo ""

# Compiler tous les fichiers Java
find src -name "*.java" -print0 | xargs -0 javac -d bin -cp src

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Compilation réussie${NC}"
else
    echo "❌ Erreur de compilation"
    exit 1
fi
echo ""

# Étape 2: Exécution du benchmark
echo -e "${BLUE}[2/3] Exécution du benchmark...${NC}"
echo -e "${YELLOW}Cela peut prendre plusieurs minutes...${NC}"
echo ""

java -cp bin BenchmarkRecherche

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Benchmark terminé${NC}"
else
    echo "❌ Erreur lors du benchmark"
    exit 1
fi
echo ""

# Étape 3: Analyse et génération des graphiques
echo -e "${BLUE}[3/3] Génération des graphiques et analyses...${NC}"
echo ""

# Vérifier si Python et les bibliothèques nécessaires sont installées
if command -v python3 &> /dev/null; then

    # Installer les dépendances si nécessaire
    echo "Vérification des dépendances Python..."
    pip3 install -q pandas matplotlib seaborn numpy 2>/dev/null || {
        echo -e "${YELLOW}Installation des bibliothèques Python nécessaires...${NC}"
        pip3 install pandas matplotlib seaborn numpy
    }

    # Exécuter le script d'analyse
    python3 analyze_results.py

    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✓ Analyse terminée${NC}"
    else
        echo "❌ Erreur lors de l'analyse"
        exit 1
    fi
else
    echo -e "${YELLOW}⚠ Python3 n'est pas installé. Les graphiques ne seront pas générés.${NC}"
    echo "Les résultats sont disponibles dans benchmark_results.csv"
fi

echo ""
echo "======================================================================"
echo -e "${GREEN}✓ ÉTUDE TERMINÉE${NC}"
echo "======================================================================"
echo ""
echo "📁 Résultats disponibles:"
echo ""
echo "  📊 Données brutes:"
echo "     - benchmark_results.csv"
echo ""
echo "  📈 Graphiques:"
echo "     - graphs/temps_execution.png"
echo "     - graphs/noeuds_explores.png"
echo "     - graphs/taux_reussite.png"
echo "     - graphs/qualite_solution.png"
echo "     - graphs/comparaison_complexite.png"
echo ""
echo "  📋 Tableaux et statistiques:"
echo "     - graphs/rapport_complet.html (OUVRIR DANS UN NAVIGATEUR)"
echo "     - graphs/tableaux_detailles.xlsx (OUVRIR AVEC EXCEL)"
echo "     - graphs/statistiques_par_algorithme.csv"
echo "     - graphs/statistiques_par_configuration.csv"
echo ""
echo "🎯 Pour voir les résultats:"
echo "  1. Ouvrez graphs/rapport_complet.html dans votre navigateur"
echo "  2. Consultez graphs/tableaux_detailles.xlsx avec Excel/LibreOffice"
echo "  3. Les graphiques PNG sont dans le dossier graphs/"
echo ""
