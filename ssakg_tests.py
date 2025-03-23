import numpy as np
import pandas as pd

from ssakg.ssakg import SSAKG
from ssakg_tester_new import SSAKG_Tester
from ssakg.utils.sequence_generator import SequenceGenerator
from theory_patterns import TheoryPatterns


def create_graph_with_sequences(graph_dim: int = 2000, sequence_length: int = 15,
                                number_sequences: int = 1200, unique_elements_in_sequence: bool = False,
                                remove_diagonals=True, weighted_edges=True) -> (
        SSAKG, np.ndarray, np.ndarray):
    sequence_min = 0
    sequence_max = graph_dim

    ssakg = SSAKG(number_of_symbols=graph_dim, sequence_length=sequence_length,
                  remove_diagonals=remove_diagonals, weighted_edges=weighted_edges, bits_graph=False)

    sequence_generator = SequenceGenerator(sequence_length=sequence_length, sequence_min=sequence_min,
                                           sequence_max=sequence_max)

    sequences = sequence_generator.generate_unique_sequences(number_sequences=number_sequences,
                                                             unique_elements=unique_elements_in_sequence)
    ssakg.insert(sequences)

    return ssakg, sequence_generator, sequences


def test_memory(number_of_symbols: int = 2000, sequence_length: int = 15, context_length: int = 6,
                number_sequences: int = 1200, unique_elements_in_sequence: bool = False) -> (SSAKG, np.ndarray):
    remove_diagonals = True
    weighted_edges = True

    ssakg, _, sequences = create_graph_with_sequences(graph_dim=number_of_symbols,
                                                      sequence_length=sequence_length,
                                                      number_sequences=number_sequences,
                                                      unique_elements_in_sequence=unique_elements_in_sequence,
                                                      remove_diagonals=remove_diagonals,
                                                      weighted_edges=weighted_edges)

    memory_comparator = SSAKG_Tester(ssakg, sequences)
    memory_comparator.make_test(context_length=context_length, show_progress=True)

    dataframe = memory_comparator.create_agreements_dataframe()
    dataframe.to_csv(f"memory_agreement_{number_of_symbols}_{sequence_length}_{context_length}.csv", index=False)

    memory_comparator.plot_agreement_histogram(draw_text=False)

    print(memory_comparator)
    print(ssakg)

    return ssakg


def test_scenes(number_of_symbols=1000, sequence_length=15, context_length=7,
                number_sequences: int = 1000,
                unique_elements_in_sequence=False):
    number_of_symbols = number_of_symbols
    sequence_length = sequence_length
    context_length = context_length
    number_sequences = number_sequences

    test_memory(number_of_symbols=number_of_symbols, sequence_length=sequence_length,
                context_length=context_length,
                number_sequences=number_sequences,
                unique_elements_in_sequence=unique_elements_in_sequence)


def test_iris():
    theory_pattern = TheoryPatterns()
    iris_accuracy = 0.1

    graph_dim = 126
    sub_graph_dim = 5
    context_length = 4
    unique_sequence = True

    number_sequences = 179

    epsilon = 0.01

    d_critical = theory_pattern.d_critical(n=graph_dim, n_f=sub_graph_dim, n_c=context_length, epsilon=epsilon,
                                           accuracy=iris_accuracy)
    print(f"Sequence length: {sub_graph_dim}")
    print(f"Context length: {context_length}")
    print(f"Critical density (theory): {d_critical:.2f}")
    s_max = theory_pattern.s_max(n=graph_dim, n_f=sub_graph_dim, n_c=context_length, epsilon=epsilon,
                                 accuracy=iris_accuracy)
    print(f"Memory capacity (theory): {s_max:.1f}")

    test_memory(number_of_symbols=graph_dim, sequence_length=sub_graph_dim,
                context_length=context_length,
                number_sequences=number_sequences,
                unique_elements_in_sequence=unique_sequence)


def table3(symbols_list: list[int], context_list: list[int], sequence_length=15,
           number_of_sequences: int = 1000, unique_elements=False, show_progress=False) -> (pd.DataFrame, str):
    dataframe_columns = [f"{context}" for context in context_list]
    dataframe_index = [f"{symbols}" for symbols in symbols_list]

    data_table = np.zeros((len(symbols_list), len(context_list)), dtype=float)

    for i, symbols in enumerate(symbols_list):
        ssakg = SSAKG(number_of_symbols=symbols, sequence_length=sequence_length)
        sequence_generator = SequenceGenerator(sequence_length=sequence_length, sequence_min=0,
                                               sequence_max=symbols)
        sequences = sequence_generator.generate_unique_sequences(number_of_sequences, unique_elements=unique_elements)

        ssakg.insert(sequences)

        ssakg_tester = SSAKG_Tester(ssakg, sequences)

        for j, context in enumerate(context_list):
            data_table[i, j] = 100 - ssakg_tester.make_test(context_length=context, show_progress=show_progress)

    caption = f"Scene recognition error in % for various context for sequence length {sequence_length}, no scenes {number_of_sequences}."
    return pd.DataFrame(data_table, index=dataframe_index, columns=dataframe_columns), caption


def table2(symbols_list: list[int], number_of_sequences_list: list[int], context_length, sequence_length=15,
           unique_elements=True, show_progress=False) -> (pd.DataFrame, str):
    dataframe_columns = [f"{number_of_sequences}" for number_of_sequences in number_of_sequences_list]
    dataframe_index = [f"{symbols}" for symbols in symbols_list]

    data_table = np.zeros((len(symbols_list), len(number_of_sequences_list)), dtype=float)

    for i, symbols in enumerate(symbols_list):
        for j, no_sequences in enumerate(number_of_sequences_list):
            ssakg = SSAKG(number_of_symbols=symbols, sequence_length=sequence_length)
            sequence_generator = SequenceGenerator(sequence_length=sequence_length, sequence_min=0,
                                                   sequence_max=symbols)
            sequences = sequence_generator.generate_unique_sequences(no_sequences, unique_elements=unique_elements)

            ssakg.insert(sequences)

            ssakg_tester = SSAKG_Tester(ssakg, sequences)
            data_table[i, j] = 100 - ssakg_tester.make_test(context_length=context_length, show_progress=show_progress)

    caption = f"Scene recognition error in % for various dataset size for sequence length {sequence_length}, context length {context_length}."
    return pd.DataFrame(data_table, index=dataframe_index, columns=dataframe_columns), caption


def table3test(unique_elements: bool):
    no_symbols_list = [1000]
    context_list = [8, 6, 5, 4, 3]
    sequence_length = 15
    number_of_sequences = 1000

    table3_dataframe, caption = table3(symbols_list=no_symbols_list, context_list=context_list,
                                       sequence_length=sequence_length,
                                       number_of_sequences=number_of_sequences, show_progress=True)
    print(caption)
    print(table3_dataframe)


def table2test(unique_elements: bool):
    no_symbols_list = [1000]
    context_length = 6
    sequence_length = 15
    number_of_sequences_list = [500, 1000, 1500, 2000, 2500, 3000]

    table2_dataframe, caption = table2(symbols_list=no_symbols_list, number_of_sequences_list=number_of_sequences_list,
                                       context_length=context_length,
                                       sequence_length=sequence_length, show_progress=True)

    print(caption)
    print(table2_dataframe)


if __name__ == "__main__":
    # test_iris()
    # test_scenes(graph_dim=1000, number_sequences=1000, sequence_length=15, context_length=6)
    # test_scenes(number_of_symbols=1000, number_sequences=1000, sequence_length=15, context_length=7)
    # test_scenes(graph_dim=615, number_sequences=1000, sequence_length=25, context_length=8)
    # test_scenes(memory_type="non_sequential", context_length=7, unique_elements_in_sequence=True)

    table3test(unique_elements=True)
    table2test(unique_elements=True)
