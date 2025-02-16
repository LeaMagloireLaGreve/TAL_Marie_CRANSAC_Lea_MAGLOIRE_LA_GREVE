
def extract_first_n_sentences(input_file, output_file, n, start=0):
    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for i, line in enumerate(infile):
            if i < start:
                continue
            if i >= start + n:
                break
            outfile.write(line)

# Train :
extract_first_n_sentences("Europarl_en.txt", "Europarl_train_100k.en", 100000)
extract_first_n_sentences("Europarl_fr.txt", "Europarl_train_100k.fr", 100000)

extract_first_n_sentences("Emea_en.txt", "Emea_train_10k.en", 10000)
extract_first_n_sentences("Emea_fr.txt", "Emea_train_10k.fr", 10000)

# Dev
extract_first_n_sentences("Europarl_en.txt", "Europarl_dev_3750.en", 3750, start=100001)
extract_first_n_sentences("Europarl_fr.txt", "Europarl_dev_3750.fr", 3750, start=100001)

# Test
extract_first_n_sentences("Europarl_en.txt", "Europarl_test_500.en", 500, start=103751)
extract_first_n_sentences("Europarl_fr.txt", "Europarl_test_500.fr", 500, start=103751)

extract_first_n_sentences("Emea_en.txt", "Emea_test_500.en", 500, start=10001)
extract_first_n_sentences("Emea_fr.txt", "Emea_test_500.fr", 500, start=10001)
