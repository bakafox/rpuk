import argparse

parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
# python3 "Lesson2/Лаба 1 (Перемножение матриц)/matrix_mult.py" -I "Lesson2/Лаба 1 (Перемножение матриц)/matrix.txt" -O "Lesson2/Лаба 1 (Перемножение матриц)/result.txt"
parser.add_argument('-I', type=str, required=True)
parser.add_argument('-O', type=str, required=True)


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
Проверка матриц на корректность размерностей (по условию),
возвращает True (можно перемножить) или False (нельзя перемножить).
Предполагается, что обе матрицы заданы корректно!
"""
def compare_matrices_dimensions(matrix1, matrix2):
    return ((len(matrix1) == len(matrix2[0]))
            and (len(matrix1[0]) == len(matrix2)))


"""
Выполняет перемножение двух переданных матриц, после чего
возвращает результат вычислений как список списков.
Предполагается, что обе матрицы заданы корректно!
"""
def multiply_matrices(matrix1, matrix2):
    result = []

    for i in range(len(matrix1)):
        result.append([])
        for j in range(len(matrix2[0])):
            result[i].append(0)
            for k in range(len(matrix2)):
                result[i][j] += matrix1[i][k] * matrix2[k][j]

    return result


"""
Выводит 1 матрицу в формате чисел, разделённых пробелами,
разделяя их пустой строкой ('\n'), в указанный файл.
Ничего не возвращает.
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
            f_output.writelines(curr_str + '\n')
        f_output.writelines('\n')


def init():
    args = parser.parse_args()
    input_path = args.I
    output_path = args.O

    [matrix1, matrix2] = parse_matrices(input_path)
    if compare_matrices_dimensions(matrix1, matrix2):
        matrixR = multiply_matrices(matrix1, matrix2)
        output_matrices(output_path, [matrixR])
        print('Матрицы успешно перемножены! Результат сохранён в файл.')
    else:
        print('Эти матрицы нельзя перемножить.')


init()
