import nltk
import spacy
from nltk.stem import WordNetLemmatizer
import os

# Télécharger les ressources pour WordNetLemmatizer (NLTK)
nltk.download('wordnet')

# Charger les modèles SpaCy
nlp_fr = spacy.load("fr_core_news_sm")  # Français
nlp_en = spacy.load("en_core_web_sm")  # Anglais

# Initialisation du lemmatiseur NLTK pour l'anglais
lemmatizer_en = WordNetLemmatizer()

def lemmatize_sentence_en(sentence):
    doc = nlp_en(sentence)  # Traitement avec SpaCy
    return ' '.join([token.lemma_ for token in doc])

def lemmatize_sentence_fr(sentence):
    doc = nlp_fr(sentence)  # Traitement avec SpaCy
    return ' '.join([token.lemma_ for token in doc])

def process_corpus(input_file, output_file, language):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            line = line.strip()
            if language == 'en':
                lemmatized_line = lemmatize_sentence_en(line)
            elif language == 'fr':
                lemmatized_line = lemmatize_sentence_fr(line)
            else:
                raise ValueError("Langue non supportée. Utilisez 'en' pour l'anglais ou 'fr' pour le français.")
            outfile.write(lemmatized_line + '\n')

def main():
    corpora = {

        'Emea_test_en': ('Emea_test_500.tok.true.clean.en', 'res/emea_test_500_lemmatized.en', 'en'),
        'Emea_test_fr': ('Emea_test_500.tok.true.clean.fr', 'res/emea_test_500_lemmatized.fr', 'fr'),

        'Emea_train_en': ('Emea_train_10k.tok.true.clean.en', 'res/emea_train_10k_lemmatized.en', 'en'),
        'Emea_train_fr': ('Emea_train_10k.tok.true.clean.fr', 'res/emea_train_10k_lemmatized.fr', 'fr'),

        'Europarl_train_en': ('Europarl_train_100k.tok.true.clean.en', 'res/europarl_train_100k_lemmatized.en', 'en'),
        'Europarl_train_fr': ('Europarl_train_100k.tok.true.clean.fr', 'res/europarl_train_100k_lemmatized.fr', 'fr'),

        'Europarl_dev_en': ('Europarl_dev_3750.tok.true.clean.en', 'res/europarl_dev_3750_lemmatized.en', 'en'),
        'Europarl_dev_fr': ('Europarl_dev_3750.tok.true.clean.fr', 'res/europarl_dev_3750_lemmatized.fr', 'fr'),

        'Europarl_test_en': ('Europarl_test2_500.tok.true.clean.en', 'res/europarl_test_500_lemmatized.en', 'en'),
        'Europarl_test_fr': ('Europarl_test2_500.tok.true.clean.fr', 'res/europarl_test_500_lemmatized.fr', 'fr')
    }
    
    for name, (input_file, output_file, lang) in corpora.items():
        if os.path.exists(input_file):
            print(f"Traitement du corpus : {name} ({lang})")
            process_corpus(input_file, output_file, lang)
            print(f"Fichier sauvegardé : {output_file}")
        else:
            print(f"Fichier {input_file} introuvable. Skipping...")

if __name__ == "__main__":
    main()
