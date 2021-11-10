#------------------------------------------------------------------------------#
# Funkcje pomocnicze dla skryptów realizujących komunikację z użytkownikiem
#
# author: Adam Kurowski
# mail:   akkurowski@gmail.com
# date:   23.10.2020
#------------------------------------------------------------------------------#

# Funkcja realizująca komunikację z użytkownikiem skryptu.
# Służy do zadania pytania z odpowiedzią tak/nie zwracaną w postaci
# zmiennej logicznej
def ask_for_user_preference(question):
    while True:
        user_input = input(question+' (t/n): ')
        if user_input == 't':
            return True
            break
        if user_input == 'n':
            return False
            break

# Funkcja służąca do komunikacji z użytkownikiem, która prosi o
# wybór jednek z kilku dostępnych opcji (poprzez podanie jej numeru).
def ask_user_for_an_option_choice(question, val_rec_prompt, items, single_choice=True):
    print(question)
    allowed_numbers = []
    for it_num, item in enumerate(items):
        print('\t (%i)'%(it_num+1)+str(item))
        allowed_numbers.append(str(it_num+1))
    while True:
        user_input  = input(val_rec_prompt)
        # Dopusczamy albo pojedyncze odpowiedzi, albo wielokrotne
        if single_choice:
            if user_input in allowed_numbers:
                return items[int(user_input)-1]
        else:
            split_input = user_input.split(' ')
            if len(split_input) == 1:
                if user_input in allowed_numbers:
                    return [items[int(user_input)-1]]
            else:
                output_options = []
                for sub_input in split_input:
                    if sub_input in allowed_numbers:
                        output_options.append(items[int(sub_input)-1])
                if len(output_options) != 0:
                    return output_options

# Funkcja do komunikacji z użytkownikiem, która prosi o podanie
# przez użytkownika wartości zmiennoprzecinkowej.
def ask_user_for_a_float(question):
    while True:
        user_input = input(question)
        try:
            float_value = float(user_input)
            return float_value
            break
        except:
            pass

# Funkcja do komunikacji z użytkownikiem, która prosi o podanie
# przez użytkownika ciągu znaków.
def ask_user_for_a_string(question):
    while True:
        user_input = input(question)
        try:
            string_val = str(user_input)
            return string_val
            break
        except:
            pass