import numpy as np


def sequence_to_bin(sequence: np.ndarray, max_symbol: int) -> str:
    max_min_length = len(bin(max_symbol)[2:])
    # max_min_length = 10

    binary_representation = ""
    for symbol in sequence:
        bin_symbol = bin(symbol)[2:].zfill(max_min_length)
        if len(bin_symbol) > max_min_length:
            print(len(bin_symbol))

        binary_representation += bin_symbol

    return binary_representation


def sequences_to_bin(sequences: np.ndarray, max_symbol: int, delimiter=None) -> list[str]:
    bin_sequences = []
    for sequence in sequences:
        bin_line = sequence_to_bin(sequence, max_symbol)
        if delimiter is not None:
            bin_line = f"{delimiter}".join(bin_line)
        bin_sequences.append(bin_line)

    return bin_sequences


def binary_string_to_sequence(binary_representation: str, no_symbols=1000) -> list:
    max_min_length = round(np.log2(no_symbols))
    decimal_sequence = []

    while len(binary_representation) > 0:
        binary_number = binary_representation[:max_min_length]
        decimal_number = int(binary_number, 2)
        decimal_sequence.append(decimal_number)
        binary_representation = binary_representation[max_min_length:]

    return decimal_sequence


def context_from_sequence_zeros_filed(context_length: int, sequence: np.ndarray) -> np.ndarray:
    context_array = np.zeros_like(sequence)
    indexes = np.random.choice(len(sequence), size=context_length, replace=False)

    context_array[indexes] = sequence[indexes]
    return context_array
