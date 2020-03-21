import numpy as np
import SLS_Algorithm
import helper_methods  as help
import model.boolean_Formel as bofo
import own_scripts.ripper_by_wittgenstein as ripper



def test_one_class_against_all():
    one_hot_vector = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])
    index_one_against_all_target = \
        np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

    index_five_against_all_target = \
        np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
    converted_one_hot_vector = help.one_class_against_all(one_hot_vector, one_class=1)
    assert np.array_equal(index_one_against_all_target, converted_one_hot_vector)

    converted_one_hot_vector = help.one_class_against_all(one_hot_vector, one_class=5)
    assert np.array_equal(index_five_against_all_target, converted_one_hot_vector)


def test_boolsche_formel_1():
    formel_1 = bofo.Boolsche_formel(np.array([255, 4, 24, 16], dtype=np.uint8),
                                    np.array([4, 4, 24, 16], dtype=np.uint8), number_of_product_term=2)

    target_output_relvant = np.array([np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0], dtype=np.uint8),
                                      np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.uint8)])
    target_output_negated = np.array([np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=np.uint8),
                                      np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.uint8)])

    target_output_formula_in_arrays_code = np.array(
        [np.array([-1, -1, -1, -1, -1, 1, -1, -1, 0, 0, 0, 0, 0, 1, 0, 0], dtype=np.int64),
         np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.int64)])

    assert np.array_equal(target_output_relvant, formel_1.pixel_relevant_in_arrays_code)
    assert np.array_equal(target_output_negated, formel_1.pixel_negated_in_arrays_code)
    assert np.array_equal(target_output_formula_in_arrays_code, formel_1.formel_in_arrays_code)
    formel_1.pretty_print_formula()
    formel_1.built_plot(0, 'Boolsche Formel from Test 1')


def test_boolsche_formel_1_Belegung():
    formel_1 = bofo.Boolsche_formel(np.array([255, 4, 24, 16], dtype=np.uint8),
                                    np.array([4, 4, 24, 16], dtype=np.uint8), number_of_product_term=2)

    Belegung = np.array([np.array([1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.bool),  # True
                         np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.bool),  # True
                         np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.bool),  # False
                         np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.bool),  # False
                         np.array([1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.bool)])  # False

    target_output = np.array([False, False, True, False, False])

    assert np.array_equal(formel_1.evaluate_belegung_like_c(Belegung), target_output)


def test_boolsche_formel_2():
    formel_2 = bofo.Boolsche_formel(np.array([255, 4], dtype=np.uint8), np.array([24, 16], dtype=np.uint8),
                                    number_of_product_term=1)
    target_output_relvant = np.array([np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0], dtype=np.uint8)])
    target_output_negated = np.array([np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.uint8)])
    target_output_formula = np.array(
        [np.array([-1, -1, -1, 1, 1, -1, -1, -1, 0, 0, 0, 0, 0, -1, 0, 0], dtype=np.int64)])

    assert np.array_equal(target_output_relvant, formel_2.pixel_relevant_in_arrays_code)
    assert np.array_equal(target_output_negated, formel_2.pixel_negated_in_arrays_code)
    assert np.array_equal(target_output_formula, formel_2.formel_in_arrays_code)
    formel_2.pretty_print_formula()
    formel_2.built_plot(0, 'Boolsche Formel from Test 2 \n ')


def creat_matrix_with_1_at_a_index(index_variable_to_fill_with_1, shape=(8, 8)):
    data = []
    label = []

    for row in range(shape[0]):
        row_data = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        if row % 2 == 0:
            row_data[index_variable_to_fill_with_1] = 1
            data.append(row_data)
            label.append(1)
        else:
            data.append(row_data)
            label.append(0)
    return np.array(data), np.array(label)


def test_transform_single_number_code_in_arrays_code():
    Boolsche_formel_object = bofo.Boolsche_formel(np.array([255], dtype=np.uint8),
                                                  # Need a object of Boolsche_formel to test methods of this class
                                                  np.array([24], dtype=np.uint8),
                                                  number_of_product_term=1)
    input_single_1 = np.array([200], dtype=np.uint8)
    input_single_2 = np.array([255], dtype=np.uint8)
    input_single_3 = np.array([0], dtype=np.uint8)
    target_output_1 = np.array([np.array([1, 1, 0, 0, 1, 0, 0, 0])])
    target_output_2 = np.array([np.array([1, 1, 1, 1, 1, 1, 1, 1])])
    target_output_3 = np.array([np.array([0, 0, 0, 0, 0, 0, 0, 0])])
    assert np.array_equal(Boolsche_formel_object.transform_number_code_in_arrays_code(input_single_1), target_output_1)
    assert np.array_equal(Boolsche_formel_object.transform_number_code_in_arrays_code(input_single_2), target_output_2)
    assert np.array_equal(Boolsche_formel_object.transform_number_code_in_arrays_code(input_single_3), target_output_3)


def test_transform_multi_number_code_in_multi_arrays_code_1():
    boolsche_formel_object_number_of_product_term_1 = bofo.Boolsche_formel(np.array([255, 4], dtype=np.uint8),
                                                                           # Need a object of Boolsche_formel to test methods of this class
                                                                           np.array([24, 16], dtype=np.uint8),
                                                                           number_of_product_term=2)
    input_multi_1 = np.array([200, 255, 0], dtype=np.uint8)

    target_multi_1 = np.array(
        [np.array([1, 1, 0, 0, 1, 0, 0, 0]), np.array([1, 1, 1, 1, 1, 1, 1, 1]), np.array([0, 0, 0, 0, 0, 0, 0, 0])])

    np.array_equal(boolsche_formel_object_number_of_product_term_1.
                   transform_number_code_in_arrays_code(input_multi_1), target_multi_1)


def test_transform_multi_number_code_in_arrays_code():
    boolsche_formel_object_number_of_product_term = bofo.Boolsche_formel(np.array([255, 4, 5, 6], dtype=np.uint8),
                                                                         np.array([24, 16, 6, 6], dtype=np.uint8),
                                                                         number_of_product_term=2)  # Need a object of Boolsche_formel to test methods of this class
    input_multi = np.array([200, 255, 0, 200], dtype=np.uint8)
    target_multi = np.array(
        [np.array([1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, ]),
         np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0])])
    np.array_equal(boolsche_formel_object_number_of_product_term.
                   transform_number_code_in_arrays_code(input_multi), target_multi)


def test_sls_algorithm_easiest():
    index_variable_to_fill_with_1 = 6
    data, label = creat_matrix_with_1_at_a_index(index_variable_to_fill_with_1, shape=(8, 8))

    found_formula = SLS_Algorithm.rule_extraction_with_sls_without_validation(data, label, 1, 1000)

    np.array_equal(label, found_formula.evaluate_belegung_like_c(data))
    found_formula.pretty_print_formula("graphische Repräsentation der extrahierten Regeln für einfachsten Datensatz")
    #found_formula.built_plot(0,"graphische Repräsentation der extrahierten Regeln für einfachsten Datensatz")


def test_sls_algorithm_1():
    data = np.array(
        [np.array([1, 1, 1, 0, 0, 0, 0, 1]),
         np.array([1, 0, 0, 0, 0, 0, 0, 1]),
         np.array([0, 0, 0, 0, 0, 0, 0, 0]),
         np.array([0, 0, 0, 0, 0, 0, 0, 1])])
    label = np.array([1,
                      0,
                      1,
                      0])

    found_formula = SLS_Algorithm.rule_extraction_with_sls_without_validation(data, label, 2, 100)
    print('Pixel_relevant_in_number_code: ', found_formula.pixel_relevant_in_number_code)
    print('Pixel_relevant_in_arrays_code: ', found_formula.pixel_relevant_in_arrays_code)
    print('pixel_negated_in_arrays_code: ', found_formula.pixel_negated_in_arrays_code)

    print('evaluate_belegung_alt', found_formula.evaluate_belegung_like_c(data))
    found_formula.pretty_print_formula('DNF found for test_sls_algorithm_1()')


def test_sls_algorithm_2():  # (x_0 and x_2)
    data = np.array(
        [np.array([1, 1, 1, 0, 1, 1, 1, 1]),
         np.array([0, 1, 1, 0, 1, 1, 1, 1]),
         np.array([0, 1, 1, 0, 1, 1, 1, 1]),
         np.array([1, 1, 1, 1, 1, 1, 1, 1])])
    label = np.array([1,
                      0,
                      0,
                      0])
    data = help.transform_to_boolean(data)
    label = help.transform_to_boolean(label)
    found_formula = SLS_Algorithm.rule_extraction_with_sls_without_validation(data, label, 1, 10000)

    print('evaluate_belegung', found_formula.evaluate_belegung_like_c(data))
    found_formula.pretty_print_formula('DNF found for test_sls_algorithm_1()')


def gen_data():
    data = np.array(
        [np.array([1, 1, 1, 0, 1, 1, 1, 1, 0]),  # True
         np.array([0, 1, 1, 0, 1, 0, 1, 1, 0]),  # False
         np.array([0, 0, 1, 0, 1, 0, 0, 1, 0]),  # False
         np.array([1, 1, 0, 1, 1, 1, 1, 1, 0]),  # True
         np.array([1, 0, 0, 0, 0, 1, 0, 0, 0]),  # True
         np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),  # False
         np.array([1, 0, 1, 0, 0, 0, 0, 0, 1]),  # True
         np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]),  # False
         np.array([1, 0, 1, 0, 0, 0, 0, 0, 1]),  # True
         np.array([0, 0, 1, 0, 1, 1, 0, 0, 0]),  # True
         np.array([1, 1, 1, 0, 1, 1, 1, 1, 0]),  # True
         np.array([0, 1, 1, 0, 1, 0, 1, 1, 0]),  # False
         np.array([0, 0, 1, 0, 1, 0, 0, 1, 0]),  # False
         np.array([1, 1, 0, 1, 1, 1, 1, 1, 0]),  # True
         np.array([1, 0, 0, 0, 0, 1, 0, 0, 0]),  # True
         np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),  # False
         np.array([1, 0, 1, 0, 0, 0, 0, 0, 1]),  # True
         np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]),  # False
         np.array([1, 0, 1, 0, 0, 0, 0, 0, 1]),  # True
         np.array([0, 0, 1, 0, 1, 1, 0, 0, 0]),  # True
         np.array([1, 1, 1, 0, 1, 1, 1, 1, 0]),  # True
         np.array([0, 1, 1, 0, 1, 0, 1, 1, 0]),  # False
         np.array([0, 0, 1, 0, 1, 0, 0, 1, 0]),  # False
         np.array([1, 1, 0, 1, 1, 1, 1, 1, 0]),  # True
         np.array([1, 0, 0, 0, 0, 1, 0, 0, 0]),  # True
         np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),  # False
         np.array([1, 0, 1, 0, 0, 0, 0, 0, 1]),  # True
         np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]),  # False
         np.array([1, 0, 1, 0, 0, 0, 0, 0, 1]),  # True
         np.array([0, 0, 1, 0, 1, 1, 0, 0, 0])  # True
         ])
    label = np.array([0,
                      0,
                      0,
                      0,
                      1,
                      0,
                      1,
                      0,
                      1,
                      1,
                      0,
                      0,
                      0,
                      0,
                      1,
                      0,
                      1,
                      0,
                      1,
                      1,
                      0,
                      0,
                      0,
                      0,
                      1,
                      0,
                      1,
                      0,
                      1,
                      1
                      ])

    return data, label

def test_sls_algorithm_3():  # (x_0 and x_2) or x_5
    data, label = gen_data()
    found_formula = SLS_Algorithm.rule_extraction_with_sls_without_validation(data, label, 2, 10000, )
    found_formula.number_of_relevant_variabels = 9

    found_formula.pretty_print_formula('DNF found for test_sls_algorithm_3()')
    np.array_equal(label, found_formula.evaluate_belegung_like_c(data))
    found_formula.built_plot(0, 'Visualisierung Formel 0 ')
    found_formula.built_plot(1, 'Visualisierung Formel 1 ')


def test_transform_to_boolean():
    input = [[1, -1, 0]]
    target_output = [[1, 0, 0]]
    np.array_equal(help.transform_to_boolean(input), target_output)


def test_data_in_kernel():
    input = np.array([[  # picture
        [[0], [1], [2], [3]],
        [[4], [5], [6], [7]],
        [[8], [9], [10], [11]],
        [[12], [13], [14], [15]]
    ]])

    padded_input = \
        [[[[0], [0], [0], [0], [0], [0]],
          [[0], [0], [1], [2], [3], [0]],
          [[0], [4], [5], [6], [7], [0]],
          [[0], [8], [9], [10], [11], [0]],
          [[0], [12], [13], [14], [15], [0]],
          [[0], [0], [0], [0], [0], [0]]]]

    target_output = [[[[0], [0]],
                      [[0], [0]]],

                     [[[0], [0]],
                      [[0], [1]]],

                     [[[0], [0]],
                      [[1], [2]]],

                     [[[0], [0]],
                      [[2], [3]]],

                     [[[0], [0]],
                      [[3], [0]]],

                     [[[0], [0]],
                      [[0], [4]]],

                     [[[0], [1]],
                      [[4], [5]]],

                     [[[1], [2]],
                      [[5], [6]]],

                     [[[2], [3]],
                      [[6], [7]]],

                     [[[3], [0]],
                      [[7], [0]]],

                     [[[0], [4]],
                      [[0], [8]]],

                     [[[4], [5]],
                      [[8], [9]]],

                     [[[5], [6]],
                      [[9], [10]]],

                     [[[6], [7]],
                      [[10], [11]]],

                     [[[7], [0]],
                      [[11], [0]]],

                     [[[0], [8]],
                      [[0], [12]]],

                     [[[8], [9]],
                      [[12], [13]]],

                     [[[9], [10]],
                      [[13], [14]]],

                     [[[10], [11]],
                      [[14], [15]]],

                     [[[11], [0]],
                      [[15], [0]]],

                     [[[0], [12]],
                      [[0], [0]]],

                     [[[12], [13]],
                      [[0], [0]]],

                     [[[13], [14]],
                      [[0], [0]]],

                     [[[14], [15]],
                      [[0], [0]]],

                     [[[15], [0]],
                      [[0], [0]]]]
    x = help.data_in_kernel(input, stepsize=2, width=2)
    np.array_equal (target_output, help.data_in_kernel(input, stepsize=1, width=2))
    
    target_output_2 = [[[[ 0],   [ 0]],  [[ 0],   [ 0]]], [[[ 0],   [ 0]],  [[ 1],   [ 2]]], [[[ 0],   [ 0]],  [[ 3],   [ 0]]], [[[ 0],   [ 4]],  [[ 0],   [ 8]]], [[[ 5],   [ 6]],  [[ 9],   [10]]], [[[ 7],   [ 0]],  [[11],   [ 0]]], [[[ 0],   [12]],  [[ 0],   [ 0]]], [[[13],   [14]],  [[ 0],   [ 0]]], [[[15],   [ 0]],  [[ 0],   [ 0]]]]
    np.array_equal (target_output_2, help.data_in_kernel(input, stepsize=2, width=2))


def test_data_in_kernel():
    input = np.array([[  # picture
        [[0], [1], [2], [3]],
        [[4], [5], [6], [7]],
        [[8], [9], [10], [11]],
        [[12], [13], [14], [15]]
    ]])

    padded_input = \
        [[[[0], [0], [0], [0], [0], [0]],
          [[0], [0], [1], [2], [3], [0]],
          [[0], [4], [5], [6], [7], [0]],
          [[0], [8], [9], [10], [11], [0]],
          [[0], [12], [13], [14], [15], [0]],
          [[0], [0], [0], [0], [0], [0]]]]

    target_output = [[[[0], [0]],
                      [[0], [0]]],

                     [[[0], [0]],
                      [[0], [1]]],

                     [[[0], [0]],
                      [[1], [2]]],

                     [[[0], [0]],
                      [[2], [3]]],

                     [[[0], [0]],
                      [[3], [0]]],

                     [[[0], [0]],
                      [[0], [4]]],

                     [[[0], [1]],
                      [[4], [5]]],

                     [[[1], [2]],
                      [[5], [6]]],

                     [[[2], [3]],
                      [[6], [7]]],

                     [[[3], [0]],
                      [[7], [0]]],

                     [[[0], [4]],
                      [[0], [8]]],

                     [[[4], [5]],
                      [[8], [9]]],

                     [[[5], [6]],
                      [[9], [10]]],

                     [[[6], [7]],
                      [[10], [11]]],

                     [[[7], [0]],
                      [[11], [0]]],

                     [[[0], [8]],
                      [[0], [12]]],

                     [[[8], [9]],
                      [[12], [13]]],

                     [[[9], [10]],
                      [[13], [14]]],

                     [[[10], [11]],
                      [[14], [15]]],

                     [[[11], [0]],
                      [[15], [0]]],

                     [[[0], [12]],
                      [[0], [0]]],

                     [[[12], [13]],
                      [[0], [0]]],

                     [[[13], [14]],
                      [[0], [0]]],

                     [[[14], [15]],
                      [[0], [0]]],

                     [[[15], [0]],
                      [[0], [0]]]]
    x = help.data_in_kernel(input, stepsize=2, width=2)
    np.array_equal(target_output, help.data_in_kernel(input, stepsize=1, width=2))

    target_output_2 = [[[[0], [0]], [[0], [0]]], [[[0], [0]], [[1], [2]]], [[[0], [0]], [[3], [0]]],
                       [[[0], [4]], [[0], [8]]], [[[5], [6]], [[9], [10]]], [[[7], [0]], [[11], [0]]],
                       [[[0], [12]], [[0], [0]]], [[[13], [14]], [[0], [0]]], [[[15], [0]], [[0], [0]]]]
    np.array_equal(target_output_2, help.data_in_kernel(input, stepsize=2, width=2))


def test_ripper_by_wittgenstein():
    data, label = gen_data()

    df = ripper.np_to_padas(data, label)
    rule_set, accuracy = ripper.wittgenstein_ripper(df, 'label', max_rules = 2)
    print('rule set: \n', rule_set, '\n accuracy:', accuracy )

    found_formula = SLS_Algorithm.rule_extraction_with_sls_without_validation(data, label, 1, 10000, )
    found_formula.number_of_relevant_variabels = 9

    found_formula.pretty_print_formula('DNF found for test_sls_algorithm_1()')

def test_extraction_with_real_data():
    Number_of_Product_term = 2
    Maximum_Steps_in_SKS = 10000

    training_set = np.load('data/real_data_for_extraction/data_for_SLS.npy')
    label_set = np.load('data/real_data_for_extraction/label_SLS.npy')
    kernel = np.load('data/real_data_for_extraction/kernel.npy')
    # bias = np.load('data/real_data_for_extraction/bias.npy')
    result_conv = np.load('data/real_data_for_extraction/result_conv.npy')
    kernel_width = kernel.shape[0]

    training_set_kernel_int = help.data_in_kernel(training_set, stepsize=2, width=kernel_width)
    #training_set_kernel = help.transform_to_boolean(training_set_kernel_int)
    sign_kernel = np.sign(kernel)
    for channel in range(1):  # label_set.shape[3]):
        label_self_calculated = help.calculate_convolution(training_set_kernel_int, sign_kernel[:, :, :, channel], result_conv)
        label_self_sign = np.sign(label_self_calculated)