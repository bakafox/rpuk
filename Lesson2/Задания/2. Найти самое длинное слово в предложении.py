inputstr = input('Введите строку для анализа: ')

words = inputstr.split()

longest_word = ''; longest_word_length = 0
for word in words:
    if len(word) > longest_word_length:
        longest_word = word
        longest_word_length = len(word)

print (f'Самое длинное слово: {longest_word} ({longest_word_length} зн.)')