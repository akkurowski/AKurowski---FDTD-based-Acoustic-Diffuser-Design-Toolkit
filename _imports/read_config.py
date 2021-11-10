#------------------------------------------------------------------------------#
# Funkcje pomocnicze do odczytu zawartości plikó konfiguracyjnych
#
# author: Adam Kurowski
# mail:   akkurowski@gmail.com
# date:   29.01.2021
#------------------------------------------------------------------------------#

from configobj import ConfigObj
import ast
import os

# Funkcja ewaluacyjna sprawdzająca, czy wyrażenia w pliku 
# konfiguracyjnym nie zawierają złośliwego kodu.
def save_eval(expression):
    try:
        return ast.literal_eval(expression)
    except:
        pass
    try:
        prs_tree = ast.parse(expression, mode="eval")
    except:
        return expression
    
    allowed_node_types = []
    allowed_node_types.append(ast.Expression)
    allowed_node_types.append(ast.Num)
    allowed_node_types.append(ast.BinOp)
    allowed_node_types.append(ast.Pow)
    allowed_node_types.append(ast.Div)
    allowed_node_types.append(ast.Mult)
    allowed_node_types.append(ast.Add)
    allowed_node_types.append(ast.List)
    allowed_node_types.append(ast.Load)
    allowed_node_types.append(ast.Mod)
    allowed_node_types.append(ast.Tuple)
    allowed_node_types.append(ast.NameConstant)
    allowed_node_types.append(ast.Name)
    allowed_node_types.append(ast.Dict)
    allowed_node_types.append(ast.Constant)
    
    all_operations_are_allowed = True
    for e in ast.walk(prs_tree):
        all_operations_are_allowed = all_operations_are_allowed and (type(e) in allowed_node_types)
        
    if all_operations_are_allowed:
        try:
            return eval(expression)
        except:
            return expression
    else:
        raise RuntimeError(f'operacja save_eval napotkała potencjalnie niebezpieczne wyrażenie do ewaluacji ({expression})')

# Procedura odczytująca i interpretująca zawartość pliku konfiguracyjnego
def read_config(config_path):
    if not os.path.isfile(config_path):
        raise RuntimeError(f'Wskazany plik konfiguracyjny ({config_path}) nie istnieje.')
    
    config = ConfigObj(config_path)

    # Interpretacja zawartości pliku konfiguracyjnego.
    settings = {}
    for section in config:
        settings.update({section:{}})
        for key in config[section]:
            value = config[section][key]
            if type(value) == list:
                for i in range(len(value)):
                    value[i] = save_eval(value[i])
            else:
                value = save_eval(value)
            settings[section].update({key:value})
    
    return settings