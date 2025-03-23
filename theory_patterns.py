import numpy as np
import math
import pandas as pd

class TheoryPatterns:
    def __init__(self):
        pass

    @staticmethod
    def ksi(n: int, n_f: int) -> float:
        # equation 2
        # n number of symbols
        # n_f number of scene objects
        return n_f * (n_f - 1) / n / (n - 1)

    @staticmethod
    def zeta(n_c: int) -> float:
        # n_c context length
        return n_c * (n_c - 1) / 2

    @staticmethod
    def s_memory_capacity(n: int, n_f: int, d: float) -> float:
        # memory capacity s pattern 3
        # n number of symbols
        # n_f number of scene objects
        ksi = TheoryPatterns.ksi(n, n_f)
        s = np.log(1 - d) / np.log(1 - ksi)
        return s

    @staticmethod
    def d_critical(n: int, n_f: int, n_c: int, epsilon: float, d_0: float = 0.5, accuracy=0.00001) -> float:
        # equation 4
        # n number of symbols
        # n_f number of scene objects
        # n_c context length
        # d_0 initial density
        # accuracy = abs(d_n-d_n_1) - desired difference between iteration steps
        ksi = TheoryPatterns.ksi(n, n_f)
        zeta = TheoryPatterns.zeta(n_c)

        d_n_1 = d_0
        d_n = np.power(-ksi * epsilon / np.log(1 - d_n_1), 1 / zeta)

        while np.abs(d_n - d_n_1) >= accuracy:
            d_n_1 = d_n
            d_n = np.power(-ksi * epsilon / np.log(1 - d_n_1), 1 / zeta)

        return d_n

    @staticmethod
    def s_max(n: int, n_f: int, n_c: int, epsilon: float, accuracy=0.00001) -> float:
        # memory capacity for critical density
        # n number of symbols
        # n_f number of scene objects
        # n_c context length

        d_critical = TheoryPatterns.d_critical(n, n_f, n_c, epsilon, d_0=0.5, accuracy=accuracy)
        s_max = TheoryPatterns.s_memory_capacity(n, n_f, d_critical)

        return s_max


def newton_symbol(n: int, k: int):
    if k < 0 or k > n:
        return 0
    return int(math.factorial(n) / (math.factorial(k) * math.factorial(n - k)))


def inhibition_error(n: int, n_f: int, n_c: int, s: int):
    # equation 26
    # memory capacity for critical density
    # n number of symbols
    # n_f number of scene objects
    # n_c context length
    # s number of scenes
    return 100 * (1 - np.pow(1 - newton_symbol(n - n_c, n_f - n_c) / newton_symbol(n, n_f), s - 1))


def inhibition_error_test(no_symbols: int, scene_length: int, context_length: int, number_of_scenes):
    return inhibition_error(no_symbols, scene_length, context_length, number_of_scenes)


def get_memory_capacity(no_symbols: int, scene_length: int, context_length: int, epsilon: float,
                        accuracy=0.0001) -> (float, str):
    memory_capacity = TheoryPatterns.s_max(no_symbols, scene_length, context_length, epsilon=epsilon, accuracy=accuracy)

    info = (f"no symbols: {no_symbols}, "
            f"scene length: {scene_length},"
            f"context: {context_length}, "
            f"memory capacity: {TheoryPatterns.s_max(no_symbols, scene_length, context_length, epsilon=epsilon, accuracy=accuracy):.0f}")

    return memory_capacity, info


def table1(symbols_list: list[int], context_list: list[int], sequence_length: int, epsilon: float, accuracy: float) -> (
        pd.DataFrame, str):
    dataframe_columns = [f"{context}" for context in context_list]
    dataframe_index = [f"{symbols}" for symbols in symbols_list]

    data_table = np.zeros((len(symbols_list), len(context_list)), dtype=float)

    for i, no_symbols in enumerate(symbols_list):
        for j, context in enumerate(context_list):
            memory_capacity, _ = get_memory_capacity(no_symbols=no_symbols, scene_length=sequence_length,
                                                     context_length=context,
                                                     epsilon=epsilon, accuracy=accuracy)
            data_table[i, j] = memory_capacity

    caption = f"Memory capacity as function of the context objects for sequence length {sequence_length}, epsilon {epsilon}."
    return pd.DataFrame(data_table, index=dataframe_index, columns=dataframe_columns), caption


def table4(symbols_list: list[int], context_list: list[int], sequence_length: int, number_of_scenes: int):
    dataframe_columns = [f"{context}" for context in context_list]
    dataframe_index = [f"{symbols}" for symbols in symbols_list]

    data_table = np.zeros((len(symbols_list), len(context_list)), dtype=float)

    for i, no_symbols in enumerate(symbols_list):
        for j, context in enumerate(context_list):
            data_table[i, j] = inhibition_error_test(no_symbols=no_symbols, scene_length=sequence_length,
                                                     context_length=context, number_of_scenes=number_of_scenes)

    caption = f"Scene recognition error in % for various context for sequence length {sequence_length}, no scenes {number_of_scenes}."
    return pd.DataFrame(data_table, index=dataframe_index, columns=dataframe_columns), caption


def get_min_context(no_symbols: int, scene_length: int, number_of_scenes, epsilon: float):
    min_context = 1
    max_context = scene_length
    f = lambda x: inhibition_error(no_symbols, scene_length, int(round(x)), number_of_scenes) - epsilon

    for i in range(min_context, max_context):
        if f(i) <= 0:
            context = i
            return context

    return None


def table5(symbols_list: list[int], scene_length_list: list[int], number_of_scenes, epsilon: float):
    dataframe_columns = [f"{sequence_length}" for sequence_length in scene_length_list]
    dataframe_index = [f"{symbols}" for symbols in symbols_list]

    data_table = np.zeros((len(symbols_list), len(scene_length_list)),dtype=object)

    for i, no_symbols in enumerate(symbols_list):
        for j, scene_length in enumerate(scene_length_list):
            context = get_min_context(no_symbols, scene_length, number_of_scenes, epsilon)
            data_table[i, j] = context

    caption = f"Required context size as a function of scene size for no scenes {number_of_scenes}, epsilon {epsilon}."
    return pd.DataFrame(data_table, index=dataframe_index, columns=dataframe_columns), caption


def table1_test():
    no_symbols_list = [615, 945, 2483]
    context_list = [3, 4, 5, 6]
    sequence_length = 25
    epsilon = 0.001
    accuracy = 0.0001

    table1_dataframe, caption = table1(no_symbols_list, context_list, sequence_length, epsilon, accuracy)
    print(caption)
    print(table1_dataframe)


def table4_test():
    context_list = [7, 6, 5, 4, 3]
    sequence_length = 25
    no_symbols_list = [615, 945, 2483]
    number_of_sequences = 1000
    table4_dataframe, caption = table4(no_symbols_list, context_list, sequence_length, number_of_sequences)
    print(caption)
    print(table4_dataframe)

def table5_test():
    no_symbols_list = [615, 945, 2483]
    sequence_length_list = [3, 10, 15, 20, 40, 45, 60, 70, 80, 90, 100]
    number_of_sequences = 1000
    epsilon = 0.001

    table5_dataframe, caption = table5(no_symbols_list, sequence_length_list, number_of_sequences, epsilon=epsilon)
    print(caption)
    print(table5_dataframe)


if __name__ == '__main__':
    # table1_test()
    # table4_test()

    table5_test()
