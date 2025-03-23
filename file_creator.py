import numpy as np

from ssakg import SSAKG, SequenceGenerator, SSAKG_Tester

from file_creator_utils import sequence_to_bin, binary_string_to_sequence

if __name__ == "__main__":
    number_of_symbols = 1000
    sequence_length = 15
    number_of_sequences = 1000

    unique_elements = True #jezeli elementy są unikatowe program nie dodaje dodatkowych symboli

    base_name = "sequences"
    file_name = base_name + ".txt"
    file_name_bin =f"{base_name}_bin.txt"

    # Jak ustawimy seed na jakąś wartość zawsze otrzymamy te same ciągi
    sequence_generator = SequenceGenerator(sequence_length=sequence_length, sequence_min=0,
                                           sequence_max=number_of_symbols, seed=10)

    # unique elements oznacza, że sekwencje nie zawierają powtarzających się elementów
    sequences = sequence_generator.generate_unique_sequences(number_of_sequences, unique_elements=unique_elements)

    # można to zapisać pliku

    np.savetxt(file_name, sequences, fmt="%d")

    # albo do pliku binarnego
    bin_sequences = []
    for sequence in sequences:
        bin_sequences.append(sequence_to_bin(sequence))

    np.savetxt("bin_sequences.txt", np.array(bin_sequences), fmt="%s")

    # plik binarny można rozkodować
    read_bin_sequences = np.genfromtxt("bin_sequences.txt", dtype=str, encoding="UTF-8")

    read_sequences = []
    for bin_sequence in read_bin_sequences:
        sequence = binary_string_to_sequence(bin_sequence)
        read_sequences.append(sequence)