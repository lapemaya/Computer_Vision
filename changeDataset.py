import os

# cartella con i file txt
cartella = "datasetGlasses2/valid/labels"

for nome_file in os.listdir(cartella):
    if nome_file.endswith(".txt"):
        percorso_file = os.path.join(cartella, nome_file)

        # leggi tutte le righe
        with open(percorso_file, "r") as f:
            righe = f.readlines()

        nuove_righe = []
        for riga in righe:
            parti = riga.strip().split()
            if len(parti) > 0:
                parti[0] = "0"  # sostituisci il primo valore
            nuove_righe.append(" ".join(parti) + "\n")

        # sovrascrivi il file con le nuove righe
        with open(percorso_file, "w") as f:
            f.writelines(nuove_righe)
