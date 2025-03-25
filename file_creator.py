import numpy as np

from ssakg import SequenceGenerator, SSAKG

from file_creator_utils import binary_string_to_sequence, sequences_to_bin, context_from_sequence_zeros_filed


def decode_file(filename: str):
    read_bin_sequences = np.genfromtxt(filename, dtype=str, encoding="UTF-8")

    read_sequences = []
    for bin_sequence in read_bin_sequences:
        sequence = binary_string_to_sequence(bin_sequence)
        read_sequences.append(sequence)


def crate_files(number_of_symbols: int, sequence_length: int, number_of_sequences: int, context_list: list[int],
                unique_elements=True,
                base_name="sequences", delimiter=None):
    file_name = f"{base_name}_no_symbols_{number_of_symbols}_no_sequences_{number_of_sequences}_length_{sequence_length}"
    file_name_bin = f"{file_name}_bin"

    sequence_generator = SequenceGenerator(sequence_length=sequence_length, sequence_min=0,
                                           sequence_max=number_of_symbols, seed=10)

    # unique elements oznacza, że sekwencje nie zawierają powtarzających się elementów
    sequences = sequence_generator.generate_unique_sequences(number_of_sequences, unique_elements=unique_elements)
    sequences += 1

    np.savetxt(file_name + ".txt", sequences, fmt="%d")
    bin_sequences = sequences_to_bin(sequences, no_symbols=number_of_symbols, delimiter=delimiter)

    np.savetxt(file_name_bin + ".txt", np.array(bin_sequences), fmt="%s")

    for context_length in context_list:
        context_file_name = f"{file_name}_context_{context_length}.txt"
        context_file_name_bin = f"{file_name}_context_{context_length}_bin.txt"

        context_array = np.empty_like(sequences)

        for i in range(len(sequences)):
            context_array[i] = context_from_sequence_zeros_filed(context_length, sequences[i])

        np.savetxt(context_file_name, np.array(context_array), fmt="%s")
        bin_sequences = sequences_to_bin(context_array, no_symbols=number_of_symbols, delimiter=delimiter)

        np.savetxt(context_file_name_bin, np.array(bin_sequences), fmt="%s")


if __name__ == "__main__":
    # Wszystkie liczby bitowe posiadają długość 10
    # Tabela 8
    crate_files(number_of_symbols=615, sequence_length=15, number_of_sequences=1000, context_list=[3],
                unique_elements=True,
                base_name=f"table8_{np.random.randint(10000)}", delimiter=",")

    # crate_files(number_of_symbols=945, sequence_length=15, number_of_sequences=100_000, context_list=[3, 4, 5, 6, 7, 8],
    #             unique_elements=True,
    #             base_name=f"table8_{np.random.randint(10000)}")
    #
    # crate_files(number_of_symbols=2483, sequence_length=15, number_of_sequences=100_000, context_list=[3, 4, 5, 6, 7, 8],
    #             unique_elements=True,
    #             base_name=f"table8_{np.random.randint(10000)}")
    #
    # # Tabela 2
    # # to jest ten największy plik najątawiej podzielić go na mniejsze w pętli
    # crate_files(number_of_symbols=2000, sequence_length=15, number_of_sequences=13_000,
    #             context_list=[6],
    #             unique_elements=True,
    #             base_name=f"table2_{np.random.randint(10000)}")
    #
    # # Tabela 3
    #
    # crate_files(number_of_symbols=615, sequence_length=15, number_of_sequences=1000,
    #             context_list=[6],
    #             unique_elements=True,
    #             base_name=f"table3_{np.random.randint(10000)}")
