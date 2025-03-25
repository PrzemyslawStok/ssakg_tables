import numpy as np


def sequence_to_bin(sequence: np.ndarray, no_symbols: int, delimiter=None) -> str:
    max_min_length = round(np.log2(no_symbols))
    binary_representation = ""
    for symbol in sequence:
        bin_symbol = bin(symbol)[2:].zfill(max_min_length)
        binary_representation += bin_symbol

    return binary_representation


def sequences_to_bin(sequences: np.ndarray, no_symbols: int, delimiter=None) -> list[str]:
    bin_sequences = []
    for sequence in sequences:
        bin_line = sequence_to_bin(sequence, no_symbols, delimiter)
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
