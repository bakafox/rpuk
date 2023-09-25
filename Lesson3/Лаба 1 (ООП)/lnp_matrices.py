"""
Ищет и парсит матрицы в заданном файле в формате чисел,
разделённых пробелами, разделённые пустыми строками ('\n'),
от начала файла. Возвращает их как список списков списков.
"""
def parse_matrices(input_path):
    matrices = []
    curr_matrix = []

    with open(input_path, 'r') as f_input:
        for input_line in f_input:
            if input_line == '\n':
                matrices.append(curr_matrix)
                curr_matrix = []
            else:
                curr_line = []
                for input_num in input_line.rstrip().split(' '):
                    curr_line.append(int(input_num))
                curr_matrix.append(curr_line)
    matrices.append(curr_matrix)

    return matrices


"""
Выполняет проверку размерностей матриц (матрицы должны
иметь одинаковые размерности, чтобы их можно было сложить),
после чего выполняет сложение двух переданных матриц
и возвращает результат вычислений как список списков.
Предполагается, что входные матрицы записаны корректно!
"""
def sum_matrices(matrix1, matrix2):
    if (not (len(matrix1) == len(matrix2))
            and (len(matrix1[0]) == len(matrix2[0]))):
        return None

    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            matrix1[i][j] += matrix2[i][j]

    return matrix1


"""
Выполняет проверку размерностей матриц (матрицы должны
иметь одинаковые размерности, чтобы их можно было вычесть),
после чего выполняет вычитание 2-й матрицы из 1-й
и возвращает результат вычислений как список списков.
Предполагается, что входные матрицы записаны корректно!
"""
def sub_matrices(matrix1, matrix2):
    if (not (len(matrix1) == len(matrix2))
            and (len(matrix1[0]) == len(matrix2[0]))):
        return None

    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            matrix1[i][j] -= matrix2[i][j]

    return matrix1


"""
Выводит матрицы в формате чисел, разделённых пробелами,
разделяя их пустой строкой ('\n'), в указанный файл
и в консоль. Ничего не возвращает.
"""
def output_matrices(output_path, matrices):
    f_output = open(output_path, 'w')

    for matrix in matrices:
        for i in range(len(matrix)):
            curr_str = ''
            for j in range(len(matrix[i])):
                if (curr_str != ''):
                    curr_str += ' '
                curr_str += str(matrix[i][j])
            print(curr_str)
            f_output.writelines(curr_str + '\n')
        f_output.writelines('\n')