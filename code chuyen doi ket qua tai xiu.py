import os

input_file = os.path.abspath("C:/Users/Mr.Nhat/Desktop/cau.txt")
output_file = os.path.abspath("C:/Users/Mr.Nhat/Desktop/cau1.txt")

def get_result(number):
    digits_sum = sum(int(d) for d in str(number))
    return "0" if digits_sum <= 10 else "1"

with open(input_file, "r", encoding="utf-16") as input_f, open(output_file, "w", encoding="utf-8") as output_f:
    for line in input_f:
        number = int(line.strip().lstrip('\ufeff'))
        result = get_result(number)
        output_f.write(result + "\n")
