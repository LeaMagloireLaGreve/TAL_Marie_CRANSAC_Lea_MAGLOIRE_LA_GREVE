# TAL_Marie_CRANSAC_Lea_MAGLOIRE_LA_GREVE
Sujet 1 : Entrainer un modèle de traduction neuronale OpenNMT à partir de scratch

README

I. Expérimentation
1. Vérification de l’installation d'Open NMT sur le corpus anglais-allemand

a. Création de l'environnement virtuel
On crée un environnement virtuel avec Anaconda avec la commande : 

'conda create --name env_opennmt python=3.9
conda init
conda activate env_opennmt'

Ensuite on installe les dépendances avec les commandes : 

pip install --upgrade gensim
pip install numpy
conda install pytorch torchvision -c pytorch
pip install OpenNMT-py

b. Téléchargement et préparation des données
On télécharge le corpus de test anglais-allemand pour la traduction automatique contenant 10 000 phrases tokenisées avec la commande : 

wget https://s3.amazonaws.com/opennmt-trainingdata/toy-ende.tar.gz
tar xf toy-ende.tar.gz
cd toy-ende

Ensuite on rajoute toy_en_de.yaml dans le dossier toy-ende.

c. Générer le vocabulaire à partir du corpus
Etape 1 : Préparation des données

On construit le vocabulaire avec la commande :

onmt_build_vocab -config toy_en_de.yaml -n_sample 10000

Résultat de la commande :
Corpus corpus_1's weight should be given. We default it to 1 for you.
[2025-02-23 11:48:29,644 INFO] Counter vocab from 10000 samples.
[2025-02-23 11:48:29,644 INFO] Build vocab on 10000 transformed examples/corpus.
[2025-02-23 11:48:30,024 INFO] Counters src: 24995
[2025-02-23 11:48:30,024 INFO] Counters tgt: 35816

Etape 2 : Entrainement du modèle

On exécute la commande suivante :

onmt_train -config toy_en_de.yaml

Résultat de la commande :
[2025-02-23 11:55:46,604 INFO] valid stats calculation
                           took: 29.44715404510498 s.
[2025-02-23 11:55:46,606 INFO] Train perplexity: 1672.87
[2025-02-23 11:55:46,606 INFO] Train accuracy: 10.7949
[2025-02-23 11:55:46,606 INFO] Sentences processed: 63454
[2025-02-23 11:55:46,606 INFO] Average bsz: 1409/1404/63
[2025-02-23 11:55:46,607 INFO] Validation perplexity: 267.466
[2025-02-23 11:55:46,607 INFO] Validation accuracy: 18.6711
[2025-02-23 11:55:46,617 INFO] Saving checkpoint toy-ende/run/model_step_1000.pt

Etape 3 : Traduction

On utilise le modèle que l’on vient d’entraîner afin de démarrer la traduction et on stocke le résultat dans le fichier toy-ende/pred_1000.txt avec la commande : 

onmt_translate -model toy-ende/run/model_step_1000.pt -src toy-ende/src-test.txt -output toy-ende/pred_1000.txt -gpu 0 -verbose

On obtient :
[2025-02-23 11:56:59,577 INFO] PRED SCORE: -1.4455, PRED PPL: 4.24 NB SENTENCES: 2737
Time w/o python interpreter load/terminate:  22.243698835372925

Afin de calculer le score BLEU, nous utilisons le script fourni : multi_bleu.pl : 

python .\src\multi_bleu.pl .\toy-ende\tgt-test.txt .\toy-ende\pred_1000.txt

On obtient : 
Reference 1st sentence: Orlando Bloom und Miranda Kerr lieben sich noch immer
MTed 1st sentence: In der <unk> der <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk>
BLEU:  0.081308593710109576
Le score BLEU obtenu est vraiment faible comparé au tableau ci-dessous : 

Score BLEU	Interprétation
< 10	Traductions presque inutiles
10 à 19	L'idée générale est difficilement compréhensible
20 à 29	L'idée générale apparaît clairement, mais le texte comporte de nombreuses erreurs grammaticales
30 à 40	Résultats compréhensibles à traductions correctes
40 à 50	Traductions de haute qualité
50 à 60	Traductions de très haute qualité, adéquates et fluides
> 60	Qualité souvent meilleure que celle d'une traduction humaine
Utilisation du moteur OpenNMT sur les corpus TRAIN, DEV et TEST

2. Utilisation du moteur Open NMT sur les corpus TRAIN, DEV et TEST

On dispose de trois corpus différents : 
- TRAIN : Europarl_train_10k.en et Europarl_train_10k.fr
- DEV : Europarl_dev_1k.en et Europarl_dev_1k.fr
- TEST : Europarl_test_500.en et Europarl_test_500.fr

Préparation du corpus

git clone https://github.com/moses-smt/mosesdecoder.git

a. Récupération du corpus EuroParl

On copie sur un répertoire /Experiments les fichiers suivants :

Europarl_dev_1k.en
Europarl_dev_1k.fr
Europarl_test_500.en
Europarl_test_500.fr
Europarl_train_10k.en
Europarl_train_10k.fr

b. Tokenisation du corpus Anglais-Français (TRAIN du corpus Europarl)
source moses.env

- Anglais (TRAIN du corpus Europarl)
../mosesdecoder/scripts/tokenizer/tokenizer.perl  -l en < Europarl_train_10k.en > Europarl_train_10k.tok.en

- Français (TRAIN du corpus Europarl)
../mosesdecoder/scripts/tokenizer/tokenizer.perl  -l fr < Europarl_train_10k.fr > Europarl_train_10k.tok.fr

Résultat:
Europarl_train_10k.tok.en
Europarl_train_10k.tok.en

c. Changement des majuscules en minuscules du corpus Anglais-Français
1. Apprentissage du modèle de transformation

- Anglais (TRAIN du corpus Europarl)
../../mosesdecoder/scripts/recaser/train-truecaser.perl --model truecase-model.en --corpus Europarl_train_10k.tok.en

- Français (TRAIN du corpus Europarl)
../../mosesdecoder/scripts/recaser/train-truecaser.perl --model truecase-model.fr --corpus Europarl_train_10k.tok.fr

Résultat:
truecase-model.en
truecase-model.fr

2. Transformation des majuscules en minuscules

- Anglais (TRAIN du corpus Europarl)
../../mosesdecoder/scripts/recaser/truecase.perl --model truecase-model.en < Europarl_train_10k.tok.en > Europarl_train_10k.tok.true.en

- Français (TRAIN du corpus Europarl)
../../mosesdecoder/scripts/recaser/truecase.perl --model truecase-model.fr < Europarl_train_10k.tok.fr > Europarl_train_10k.tok.true.fr

Résultat:
Europarl_train_10k.tok.true.en
Europarl_train_10k.tok.true.fr

d. Nettoyage en limitant la longueur des phrases à 80 caractères (TRAIN du corpus Europarl)

- Anglais (corpus Europarl)
../../mosesdecoder/scripts/training/clean-corpus-n.perl Europarl_train_10k.tok.true fr en Europarl_train_10k.tok.true.clean 1 80

- Français (corpus Europarl)
../../mosesdecoder/scripts/training/clean-corpus-n.perl Europarl_train_10k.tok.true en fr Europarl_train_10k.tok.true.clean 1 80

Résultat :
Europarl_train_10k.tok.true.clean.en (wc -l Europarl_train_10k.tok.true.clean.en => 9767 Europarl_train_10k.tok.true.clean.en)

Europarl_train_10k.tok.true.clean.fr (wc -l Europarl_train_10k.tok.true.clean.fr => 9767 Europarl_train_10k.tok.true.clean.fr)

3. Générer le vocabulaire à partir du corpus

On créé le fichier de configuration europarl.yaml et on l'éxécute avec les mêmes commandes que pour le corpus anglais-allemand en modifiant seulement les paramètres : 

onmt_build_vocab -config europarl.yaml -n_sample 10000

On obtient : 
Corpus corpus_1's weight should be given. We default it to 1 for you.
[2025-02-12 15:00:08,531 INFO] Counter vocab from 10000 samples.
[2025-02-12 15:00:08,531 INFO] Build vocab on 10000 transformed examples/corpus.
[2025-02-12 15:00:08,881 INFO] Counters src: 11547
[2025-02-12 15:00:08,881 INFO] Counters tgt: 14694

Etape 2 : Entraînement du modèle

onmt_train -config europarl.yaml

On obtient : 
[2025-02-12 15:16:56,850 INFO] Train perplexity: 319.744
[2025-02-12 15:16:56,850 INFO] Train accuracy: 15.2171
[2025-02-12 15:16:56,851 INFO] Sentences processed: 59331
[2025-02-12 15:16:56,851 INFO] Average bsz: 1826/1762/64
[2025-02-12 15:16:56,851 INFO] Validation perplexity: 834.961
[2025-02-12 15:16:56,851 INFO] Validation accuracy: 9.61589
[2025-02-12 15:16:56,854 INFO] Saving checkpoint europarl/run/model_step_1000.pt

Etape 3 : Traduction

onmt_translate -model europarl/run/model_step_500.pt -src europarl/Europarl_test_500.en -output europarl/Europarl_res_500.fr -gpu 0 -verbose

On obtient : 
[2025-02-14 17:21:35,885 INFO] PRED SCORE: -1.3273, PRED PPL: 3.77 NB SENTENCES: 500
Time w/o python interpreter load/terminate:  4.892900466918945

Etape 4 : Calcul du score BLEU

python MT-Evaluation/BLEU/compute-bleu.py europarl/Europarl_test_500.fr europarl/Europarl_res_500.fr

On obtient : Un score BLEU de : BLEU: 1.140900862053107%
Le score BLEU est très faible, cela étant dû au fait que les corpus n'ont pas beaucoup de phrases.

II. Evaluation sur des corpus parallèles en formes fléchies à large échelle
1. Réalisation des expérimentations sur deux corpus parallèles pour le couple de langues anglais-français.

On récupère les textes Europarl et Emea .txt sur : 
https://opus.nlpl.eu/Europarl/fr&en/v8/Europarl
https://opus.nlpl.eu/EMEA/fr&en/v3/EMEA 

On commence par utiliser le script script/decoupage.py afin de suivre les indications suivantes : 

Corpus d’apprentissage (TRAIN ) : 
- 100K (Europarl) : Europarl_train_100k.en, Europarl_train_100k.fr ➔ Il faut prendre les 100 000 premières phrases du corpus.
- 10K (Emea) : Emea_train_10k.en, Emea_train_10k.fr ➔ Il faut prendre les 10 000 premières phrases du corpus.

Corpus de développement (DEV ) : 
- Europarl_dev_3750.en ➔ Il faut prendre 3750 phrases à partir du corpus Europarl en commençant par la phrase au rang 100 001
- Europarl_dev_3750.fr ➔ Il faut prendre 3750 phrases à partir du corpus Europarl en commençant par la phrase au rang 100 001

Corpus de test (TEST ) :
- Europarl : Europarl_test_500.en, Europarl_test_500.fr ➔ Il faut prendre 500 phrases à partir du corpus Europarl en commençant par la phrase au rang 103751
- Emea : Emea_test_500.en, Emea_test_500.fr ➔ Il faut prendre 500 phrases à partir du corpus Emea en commençant par la phrase au rang 10001

python script/decoupage.py

Ensuite on recommence les même étapes avec les scripts de mosesdecoder afin de tokeniser et de nettoyer les corpus : 
Tokenisation : 
mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < Europarl_train_100k.en > Europarl_train_100k.tok.en
mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < Europarl_dev_3750.en > Europarl_dev_3750.tok.en
mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < Europarl_test2_500.en > Europarl_test2_500.tok.en
mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < Emea_train_10k.en > Emea_train_10k.tok.en
mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < Emea_test_500.en > Emea_test_500.tok.en

mosesdecoder/scripts/tokenizer/tokenizer.perl -l fr < Europarl_train_100k.fr > Europarl_train_100k.tok.fr
mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < Europarl_dev_3750.fr > Europarl_dev_3750.tok.fr
mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < Europarl_test2_500.fr > Europarl_test2_500.tok.fr
mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < Emea_train_10k.fr > Emea_train_10k.tok.fr
mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < Emea_test_500.fr > Emea_test_500.tok.fr

True caser : 
mosesdecoder/scripts/recaser/train-truecaser.perl --model truecase-model.en --corpus Europarl_train_100k.tok.en
mosesdecoder/scripts/recaser/train-truecaser.perl --model truecase-model.en --corpus Emea_train_10k.tok.en

mosesdecoder/scripts/recaser/train-truecaser.perl --model truecase-model.fr --corpus Europarl_train_100k.tok.fr
mosesdecoder/scripts/recaser/train-truecaser.perl --model truecase-model.fr --corpus Emea_train_10k.tok.fr

mosesdecoder/scripts/recaser/truecase.perl --model truecase-model.en < Europarl_train_100k.tok.en > Europarl_train_100k.tok.true.en
mosesdecoder/scripts/recaser/truecase.perl --model truecase-model.en < Europarl_dev_3750.tok.en > Europarl_dev_3750.tok.true.en
mosesdecoder/scripts/recaser/truecase.perl --model truecase-model.en < Europarl_test2_500.tok.en > Europarl_test2_500.tok.true.en
mosesdecoder/scripts/recaser/truecase.perl --model truecase-model.en < Emea_train_10k.tok.en > Emea_train_10k.tok.true.en
mosesdecoder/scripts/recaser/truecase.perl --model truecase-model.en < Emea_test_500.tok.en > Emea_test_500.tok.true.en

mosesdecoder/scripts/recaser/truecase.perl --model truecase-model.fr < Europarl_train_100k.tok.fr > Europarl_train_100k.tok.true.fr
mosesdecoder/scripts/recaser/truecase.perl --model truecase-model.fr < Europarl_dev_3750.tok.fr > Europarl_dev_3750.tok.true.fr
mosesdecoder/scripts/recaser/truecase.perl --model truecase-model.fr < Europarl_test2_500.tok.fr > Europarl_test2_500.tok.true.fr
mosesdecoder/scripts/recaser/truecase.perl --model truecase-model.fr < Emea_train_10k.tok.fr > Emea_train_10k.tok.true.fr
mosesdecoder/scripts/recaser/truecase.perl --model truecase-model.fr < Emea_test_500.tok.fr > Emea_test_500.tok.true.fr

Nettoyage : 
mosesdecoder/scripts/training/clean-corpus-n.perl Europarl_train_100k.tok.true fr en Europarl_train_100k.tok.true.clean 1 80
mosesdecoder/scripts/training/clean-corpus-n.perl Europarl_dev_3750.tok.true fr en Europarl_dev_3750.tok.true.clean 1 80
mosesdecoder/scripts/training/clean-corpus-n.perl Europarl_test2_500.tok.true fr en Europarl_test2_500.tok.true.clean 1 80
mosesdecoder/scripts/training/clean-corpus-n.perl Emea_train_10k.tok.true fr en Emea_train_10k.tok.true.clean 1 80
mosesdecoder/scripts/training/clean-corpus-n.perl Emea_test_500.tok.true fr en Emea_test_500.tok.true.clean 1 80

English : 
Europarl_train_100k.tok.true.clean.en (95013 lignes avec la commande wc -l)
Europarl_dev_3750.tok.true.clean.en (3622 lignes avec la commande wc -l)
Europarl_test2_500.tok.true.clean.en (500 lignes avec la commande wc -l)
Emea_train_10k.tok.true.clean.en (8612 lignes avec la commande wc -l)
Emea_test_500.tok.true.clean.en (500 lignes avec la commande wc -l)

French : 
Europarl_train_100k.tok.true.clean.fr (95013 lignes avec la commande wc -l)
Europarl_dev_3750.tok.true.clean.fr (3622 lignes avec la commande wc -l)
Europarl_test2_500.tok.true.clean.fr (500 lignes avec la commande wc -l)
Emea_train_10k.tok.true.clean.fr (8612 lignes avec la commande wc -l)
Emea_test_500.tok.true.clean.fr (500 lignes avec la commande wc -l)

2. Apprentissage avec OpenNMT

a. Apprentissage de la run n°1

On commence avec la run n°1 en utilisant Europarl_train_100K pour l’apprentissage (nombre de phrases) et Europarl_dev_3750 pour le tuning (nombre de phrases). On configure le fichier formes_flechies1.yaml pour se faire. Ensuite, on génère le vocabulaire avec la commande suivante :

onmt_build_vocab -config formes_flechies1.yaml -n_sample 100000

On obtient : 
95013 europarl/Europarl_train_100k.tok.true.clean.en
Corpus corpus_1's weight should be given. We default it to 1 for you.
[2025-02-23 13:33:14,925 INFO] Counter vocab from 100000 samples.
[2025-02-23 13:33:14,926 INFO] Build vocab on 100000 transformed examples/corpus.
[2025-02-23 13:33:18,714 INFO] Counters src: 33573
[2025-02-23 13:33:18,714 INFO] Counters tgt: 40718

onmt_train -config formes_flechies1.yaml

took: 19.427539110183716 s.
[2025-02-23 14:03:35,973 INFO] Train perplexity: 69.9261
[2025-02-23 14:03:35,973 INFO] Train accuracy: 24.377
[2025-02-23 14:03:35,973 INFO] Sentences processed: 639780
[2025-02-23 14:03:35,973 INFO] Average bsz: 1665/1864/64
[2025-02-23 14:03:35,973 INFO] Validation perplexity: 70.1556
[2025-02-23 14:03:35,973 INFO] Validation accuracy: 25.9583
[2025-02-23 14:03:35,979 INFO] Saving checkpoint formes_flechies1/model_step_10000.pt

b. Apprentissage de la run n°2

On fait exactement pareil avec la run n°2 en utilisant cette fois-ci Europarl_train_100K et Emea_train_10k pour l’apprentissage et Europarl_dev_3750 pour le tuning. On configure ici le fichier formes_flechies2.yaml. Ensuite, on génère le vocabulaire avec la commande suivante :

onmt_build_vocab -config formes_flechies2.yaml -n_sample 100000

Puis, on démarre l’apprentissage :

onmt_train -config formes_flechies2.yaml

3. Traduction et évaluation du score BLEU

a. Traduction avec le modèle de la run n°1

On va réaliser deux traductions avec deux corpus de test différents : un du domain et un hors-domaine.

Traduction du corpus du domaine avec le corpus Europarl_test2_500.tok.true.clean.en :

onmt_translate -model formes_flechies1/model_step_10000.pt -src europarl/Europarl_test2_500.tok.true.clean.en -output formes_flechies1/pred_domaine.txt -verbose

[2025-02-19 12:46:15,794 INFO] PRED SCORE: -1.2579, PRED PPL: 3.52 NB SENTENCES: 478
Time w/o python interpreter load/terminate:  5.586419582366943

Traduction du corpus hors-domaine avec le corpus Emea_test_500.tok.true.clean.en :

onmt_translate -model formes_flechies1/model_step_10000.pt -src EMEA/Emea_test_500.tok.true.clean.en -output formes_flechies1/pred_hors_domaine.txt -verbose

[2025-02-19 12:47:05,698 INFO] PRED SCORE: -1.2181, PRED PPL: 3.38 NB SENTENCES: 432
Time w/o python interpreter load/terminate:  4.273453712463379

Ensuite nous calculons le score BLEU pour chaque corpus.

Calcul du score BLEU pour le corpus du domaine :

python script/compute-bleu.py europarl/Europarl_test2_500.tok.true.clean.fr formes_flechies1/pred_domaine.txt

BLEU :  25,82940209302392039%

Calcul du score BLEU pour le corpus hors-domaine :

python script//compute-bleu.py EMEA/Emea_test_500.tok.true.clean.fr formes_flechies1/pred_hors_domaine.txt

BLEU :  0.127029399330393927%

Ces résultats montrent que le modèle formé uniquement avec Europarl (run n°1) offre de meilleures performances pour le domaine général (BLEU = 25,82), mais il est largement sous-performant pour le corpus hors-domaine (BLEU = 0,12). Au contraire, l’ajout de EMEA dans la run n°2 améliore considérablement la performance pour les données hors-domaine (BLEU = 69,17), mais au prix de la perte de qualité sur le domaine général (BLEU = 14,36).

Aussi, lors de nos expérimentations avec OpenNMT, nous avons étudié l’impact du paramètre train_steps, qui définit le nombre d’étapes d’apprentissage du modèle. Pour cela, nous avons comparé deux modèles entraînés avec des valeurs différentes : l’un avec 1 000 étapes et l’autre avec 10 000 étapes. L’évaluation de ces modèles à l’aide du score BLEU a révélé une différence significative en termes de performance. 

En effet, le modèle entraîné sur 10 000 étapes a obtenu un score BLEU de 25,82, tandis que celui entraîné sur 1 000 étapes n’a atteint que 4,33. Cette comparaison met en évidence l’importance du nombre d’étapes d’apprentissage : un nombre trop faible ne permet pas au modèle de bien généraliser et d’apprendre des représentations suffisamment riches, tandis qu’un entraînement plus long permet une meilleure adaptation aux données et une amélioration des performances en traduction.

Ainsi, ces résultats mesurés illustrent le phénomène de spécialisation contre la généralisation en traduction automatique : un modèle entraîné sur un seul domaine est performant sur ce dernier mais échoue sur d'autres (run n°1), tandis qu'un modèle plus généraliste, bien que plus polyvalent, peut perdre en précision sur son domaine d’origine (run n°2), comme nous l’avons vu avec les expérimentations ci-dessus avec Europarl. 

b. Traduction avec le modèle de la run n°2

Ici, on effectue les mêmes commandes que pour la run n°1, mais avec le modèle de la run n°2.

Traduction du corpus du domaine avec le corpus Europarl_test2_500.tok.true.clean.en :

onmt_translate -model data/formes_flechies2/model_step_10000.pt -src data/europarl/Europarl_test2_500.tok.true.clean.en -output data/formes_flechies2/pred_domaine.txt -verbose

Traduction du corpus hors-domaine avec le corpus Emea_test_500.tok.true.clean.en :

onmt_translate -model data/formes_flechies2/model_step_10000.pt -src data/EMEA/Emea_test_500.tok.true.clean.en -output data/formes_flechies2/pred_hors_domaine.txt -verbose

Ensuite nous calculons le score BLEU pour chaque corpus.

Calcul du score BLEU pour le corpus du domaine :

python ./src/compute-bleu.py ./europarl/Europarl_test2_500.tok.true.clean.fr ./formes_flechies2/pred_domaine.txt
BLEU :  14.36109283738290393%

Calcul du score BLEU pour le corpus hors-domaine :

python ./src/compute-bleu.py ./EMEA/Emea_test_500.tok.true.clean.fr ./formes_flechies2/pred_hors_domaine.txt
BLEU :  69.17259609402039404%

4. Evaluation sur des corpus parallèles en lemmes à large échelle

a. Lemmatisation des corpus

On crée un script python script/lemmatization.py permettant de lemmatiser un texte.

Il faut utiliser les commandes pour installer les librairies de lemmatisation anglais et français : 

pip install nltk
pip install nltk spacy
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
pip install git+https://github.com/ClaudeCoulombe/FrenchLefffLemmatizer.git

Lemmatisation des corpus anglais et français :

python lemmatisation.py

b. Apprentissage avec OpenNMT
On recommence les phases d’apprentissage en suivant le même protocole que précédemment.
Nous avons créé les fichiers de configurations lemmes1.yaml et lemmes2.yaml afin d’utiliser les nouveaux corpus générés lors de l’étape précédente.

Génération du vocabulaire pour chaque run :

onmt_build_vocab -config lemmes1.yaml -n_sample 100000
onmt_build_vocab -config lemmes2.yaml -n_sample 100000

(env_opennmt) secours@a-201570:~$ onmt_build_vocab -config lemmes1.yaml -n_sample 100000
Corpus corpus_1's weight should be given. We default it to 1 for you.
[2025-02-19 12:21:57,910 INFO] Counter vocab from 100000 samples.
[2025-02-19 12:21:57,910 INFO] Build vocab on 100000 transformed examples/corpus.
[2025-02-19 12:22:00,783 INFO] Counters src: 24354
[2025-02-19 12:22:00,784 INFO] Counters tgt: 26867

(env_opennmt) secours@a-201570:~$ onmt_build_vocab -config lemmes2.yaml -n_sample 100000
Corpus corpus_1's weight should be given. We default it to 1 for you.
Corpus corpus_2's weight should be given. We default it to 1 for you.
[2025-02-19 12:23:35,062 INFO] Counter vocab from 100000 samples.
[2025-02-19 12:23:35,062 INFO] Build vocab on 100000 transformed examples/corpus.
[2025-02-19 12:23:38,168 INFO] Counters src: 26239
[2025-02-19 12:23:38,168 INFO] Counters tgt: 28930

Lancement des apprentissages :

onmt_train -config lemmes1.yaml
onmt_train -config lemmes2.yaml

c. Traduction et évaluation du score BLEU

Traduction avec le modèle de la run n°1

On effectue 2 traductions : l’un avec un corpus de test du domaine, et un autre hors-domaine.

Traduction du corpus du domaine :
onmt_translate -model lemmes1/model_step_10000.pt -src res/europarl_test_500_lemmatized.en -output lemmes1/pred_domaine.txt -verbose

[2025-02-19 14:45:44,641 INFO] PRED SCORE: -1.0429, PRED PPL: 2.84 NB SENTENCES: 478
Time w/o python interpreter load/terminate:  9.095455646514893

Traduction du corpus hors-domaine :

onmt_translate -model lemmes1/model_step_10000.pt -src res/emea_test_500_lemmatized.en -output lemmes1/pred_hors_domaine.txt -verbose

[2025-02-19 15:01:49,357 INFO] PRED SCORE: -0.9912, PRED PPL: 2.69 NB SENTENCES: 432
Time w/o python interpreter load/terminate:  5.187250375747681

Ensuite nous calculons le score BLEU pour chaque corpus.

Calcul du score BLEU pour le corpus du domaine :

python script/compute-bleu.py res/europarl_test_500_lemmatized.fr lemmes1/pred_domaine.txt

SCORE BLEU :  18,671928883920022%


Calcul du score BLEU pour le corpus hors-domaine :

python script/compute-bleu.py res/europarl_test_500_lemmatized.fr lemmes1/pred_hors_domaine.txt

SCORE BLEU :  0.36125621452920918%

Traduction avec le modèle de la run n°2

On effectue les mêmes commandes que pour la run n°1, mais avec le modèle de la run n°2. 

Traduction du corpus du domaine :

onmt_translate -model lemmes2/model_step_10000.pt -src res/europarl_test_500_lemmatized.en -output lemmes2/pred_domaine.txt -verbose

Traduction du corpus hors-domaine :

onmt_translate -model lemmes2/model_step_10000.pt -src res/europarl_test_500_lemmatized.en -output lemmes2/pred_hors_domaine.txt -verbose

Ensuite nous calculons le score BLEU pour chaque corpus.

Calcul du score BLEU pour le corpus du domaine :

python script/compute-bleu.py europarl/test_500_lemmatized.fr lemmes2/pred_domaine.txt

SCORE BLEU : 9.9342883998403%

Calcul du score BLEU pour le corpus hors-domaine :

python script/compute-bleu.py EMEA/test_500_lemmatized.fr lemmes2/pred_hors_domaine.txt

SCORE BLEU : 77.3792881992883%

Auteurs : Marie CRANSAC et Léa MAGLOIRE LA GREVE 
