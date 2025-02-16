import sys
import nltk
import re
from nltk.stem import WordNetLemmatizer
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer

# Télécharger les ressources nécessaires pour NLTK
nltk.download('wordnet')

# Initialisation des lemmatiseurs
lemmatizer_en = WordNetLemmatizer()
lemmatizer_fr = FrenchLefffLemmatizer()

# Vérification des arguments
if (len(sys.argv) != 4):
    raise ValueError("Il faut que les arguments soient : python lemmatisation.py [en/fr] input_file output_file")

language = sys.argv[1] # récupérer la langue en argument
input_file = sys.argv[2] # récupérer le fichier d'entrée en argument
output_file = sys.argv[3] # récupérer le fichier de sortie en argument

def lemmatize_file(language, input_file, output_file):
    """Lemmatisation du contenu d'un fichier en fonction de la langue."""
    if language == 'en':
        lemmatizer = lemmatizer_en
    elif language == 'fr':
        lemmatizer = lemmatizer_fr
    else:
        raise ValueError("Langue non prise en charge. Utilisez 'en' ou 'fr'")
    with open(input_file, 'r', encoding="utf-8") as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            line = line.strip()  # Supprimer les espaces inutiles
            words = re.findall(r'\b\w+\b', line) # Extraire les mots de la ligne
            lemmatized_words = [lemmatizer.lemmatize(word) for word in words]  # Lemmatiser les mots en minuscule
            lemmatized_line = ' '.join(lemmatized_words)  # Joindre les mots en une ligne
            f_out.write(lemmatized_line + '\n')  # Écrire la ligne lemmatisée dans le fichier de sortie
    print(f"Lemmatisation terminée. Résultat enregistré dans {output_file}")

lemmatize_file(language, input_file, output_file)