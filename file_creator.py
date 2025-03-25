import numpy as np

from ssakg import SequenceGenerator, SSAKG

from file_creator_utils import binary_string_to_sequence, sequences_to_bin


def decode_file(filename: str):
    read_bin_sequences = np.genfromtxt(filename, dtype=str, encoding="UTF-8")

    read_sequences = []
    for bin_sequence in read_bin_sequences:
        sequence = binary_string_to_sequence(bin_sequence)
        read_sequences.append(sequence)


def crate_files(number_of_symbols: int, sequence_length: int, number_of_sequences: int, context_list: list[int],
                unique_elements=True,
                base_name="sequences", delimiter=","):
    file_name = f"{base_name}_no_symbols_{number_of_symbols}_no_sequences_{number_of_sequences}_length_{sequence_length}"
    file_name_bin = f"{file_name}_bin"

    sequence_generator = SequenceGenerator(sequence_length=sequence_length, sequence_min=0,
                                           sequence_max=number_of_symbols, seed=10)

    # unique elements oznacza, że sekwencje nie zawierają powtarzających się elementów
    sequences = sequence_generator.generate_unique_sequences(number_of_sequences, unique_elements=unique_elements)

    np.savetxt(file_name + ".txt", sequences, fmt="%d")
    bin_sequences = sequences_to_bin(sequences, no_symbols=number_of_symbols, delimiter=delimiter)

    np.savetxt(file_name_bin + ".txt", np.array(bin_sequences), fmt="%s")

    for context_length in context_list:
        context_file_name = f"{file_name}_context_{context_length}.txt"
        context_file_name_bin = f"{file_name}_context_{context_length}_bin.txt"

        context_array = np.empty([number_of_sequences, context_length], dtype=int)

        for i in range(len(sequences)):
            context_array[i] = SSAKG.context_from_sequence(context_length, sequences[i])

        np.savetxt(context_file_name, np.array(bin_sequences), fmt="%s")
        bin_sequences = sequences_to_bin(context_array, no_symbols=number_of_symbols, delimiter=delimiter)

        np.savetxt(context_file_name_bin, np.array(bin_sequences), fmt="%s")


if __name__ == "__main__":
    crate_files(number_of_symbols=1000, sequence_length=15, number_of_sequences=1000, context_list=[3, 4, 5],
                unique_elements=True,
                base_name=f"sequences_{np.random.randint(10000)}")
