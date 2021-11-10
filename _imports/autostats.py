import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
from scikit_posthocs import posthoc_dunn
from scipy.stats import levene,shapiro, kruskal, ttest_1samp, chisquare
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Procedura aplikująca jednocześnie wiele poprawek na wielokrotne testowanie

# Rozwinięcia skrótów i opis działania funkcji aplikującej poprawki:
# https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html
# https://en.wikipedia.org/wiki/Holm%E2%80%93Bonferroni_method
# https://en.wikipedia.org/wiki/False_discovery_rate
# https://www.researchgate.net/post/What_is_your_prefered_p-value_correction_for_multiple_tests

def apply_multiple_multitest_corrections(p_val_tab, significance_level=0.05, save_excel = False):
    # Ogólna zasada jest taka, konserwatywna poprawka  - wyniki są pewne za cenę nieco podwyższonego ryzyka fałszywych pozytywów
    # (w porównaniu do innych poprawek - np. Simesa-Hochberga lub Benjamini-Hochberga).
    # Poprawka niekonserwatywna - tak jak np. wspomniana poprawka Simesa-Hochberga, mamy więcej "odkryć"
    # ale kosztem większego ryzyka posiadania "fałszywego pozytywu"

    # Z wykonywanych poprawek najbardziej konserwatywna jest poprawka Holma-Sidaka, Simesa-Hochberga jest umiarkowana, 
    # najbardziej "liberalna" i najchętniej przepuszczająca wyniki jest poprawka Benjamini-Hochberga.
    # poprawka Benjamini-Hochberga oznaczona jest skrótem fdr_bh
    
    CORRECTION_METHODS = ['holm-sidak','simes-hochberg','fdr_bh']
    
    def apply_correction(p_values, sig_level, correction_method):
        reject_v, p_vals_corrected,_,_ = multipletests(p_values,alpha=sig_level,method=correction_method)
        
        out_tab = []
        for p_val_orig, p_val_corr, is_rejected in zip(p_values, p_vals_corrected, reject_v):
            rejection_text = 'nie'
            if is_rejected: rejection_text = 'tak'
            out_tab.append({'Oryginalna p-wartość':'%.3f'%p_val_orig,'Wartość po korekcji':'%.3f'%p_val_corr,f'Czy hipoteza zerowa jest odrzucona? (alpha = {sig_level})':rejection_text})
        out_tab = pd.DataFrame(out_tab)
        return out_tab

    if save_excel:
        # zapis danych do wynikowego pliku Excela
        writer = pd.ExcelWriter('wyniki skorygowane.xlsx')
    for correction_name in CORRECTION_METHODS:
        data_df = apply_correction(p_val_tab, significance_level, correction_name)
        
        print('------------------------------------------------------------------------------')
        print(f'Wyniki zaaplikowania poprawki {correction_name} na wielokrotne testowanie:')
        print(data_df, '\n')
        
        if save_excel:
            data_df.to_excel(writer, correction_name, index=False)
    if save_excel:
        writer.save()
        writer.close()
        print('Wyniki po poprawkach zapisane do pliku Excela.')

# Czasem potrzebne jest "rozszycie" DataFrame'a do listy danych z każdej kolumny osobno
def df_to_collist(df):
    output = []
    for cn in df.columns:
        output.append(np.array(df[cn].dropna()))
    return output

# Procedura automatycznie aplikująca pipeline testowania bazujący na testach ANOVA/Kruskala-Wallisa
def autotest(input_data, significance_level=0.05):
    # Śledzenie informacji o preprocessingu do raportowania
    diary_info = {}
    
    # domyślnie zakładamy, że skorzystamy z testu ANOVA
    # zrezygnujemy z niej na rzecz testu Kruskala-Wallisa, jeżeli
    # wstępne testy wykażą brak spełnienia założeń do jej wykorzystania
    use_ANOVA = True
    
    if type(input_data) == dict:
        data_df = pd.DataFrame.from_dict(input_data, orient='index').T
    elif type(input_data) == pd.DataFrame:
        data_df = input_data
    else:
        raise RuntimeError('Funkcja autotest przyjmuje dane jedynie jako słownik kolumn lub DataFrame.')

    print('-----')
    print('Testy na różnice pomiędzy wynikami poszczególnych grup')
    print()
    print('Test Levene\'a na równość wariancji grup (wyników list):')
    print('(jeśli jest równość - przeprowadzamy test ANOVA, jeśli nie - Kruskala-Wallisa)')
    result_levene = levene(*df_to_collist(data_df))
    print('\tstatystyka testu Levene\'a: %2.2f'%(result_levene.statistic))
    print('\tp-wartość testu Levene\'a: %2.2f'%(result_levene.pvalue))
    
    diary_info.update({'result_Levene':result_levene})
    
    print()
    print('WYNIK TESTU NA RÓWNOŚĆ WARIANCJI: ',end='')
    if result_levene.pvalue<significance_level:
        print('wariancje nie są równe - musi być przeprowadzony test Kruskala-Wallisa')
        use_ANOVA = False
    else:
        print('wariancje są równe - można przeprowadzić test ANOVA pod warunkiem, że pozwolą na to testy Shapiro-Wilka')

    print()

    # Testy Shapiro-Wilka wykonujemy tylko jeżeli mamy równość wariancji (homoskedastyczność)
    if use_ANOVA:
        print('Testy Shapiro-Wilka z poprawką na wielokrotne testowanie:')
        print('(jeśli wszędzie jest normalność - przeprowadzamy test ANOVA, jeśli nie - Kruskala-Wallisa)')
        print()
        shapirowilk_pvals = []
        # Każdą grupę testujemy na normalność osobno, więc powtarzamy testy.
        # Powtarzając testy zawsze ryzykujemy pomylenie się i np. uznanie że
        # dana grupa ma rozkład normalny przez pomyłkę (każdy test ma losową szansę
        # na pomylenie się). Dlatego pakietem multipletests aplikujemy do p-wartości
        # testu Shapiro-Wilka poprawkę, która minimalizuje szansę na losowe pomylenie
        # się przy testowaniu na to, czy którakolwiek z grup przypadkiem nie ma 
        # niegaussowkiego rozkładu wartości
        for smplt_v in df_to_collist(data_df):
            sw_stat, sw_pval = shapiro(smplt_v)
            shapirowilk_pvals.append(sw_pval)
        reject_v, p_vals,_,_ = multipletests(shapirowilk_pvals,alpha=significance_level)
        print('p-wartości testów Shapiro-Wilka po korekcji na wielokrotne testowanie:')
        for val in p_vals: print('%2.2f'%val, end=' ')
        diary_info.update({'result_corrected_Shapiro-Wilk':p_vals})
        print()

        print()
        print('WYNIK TESTU NA NORMALNOŚĆ GRUP: ',end='')
        if np.any(reject_v):
            print('przynajmniej jedna grupa nie ma normalnego rozkładu wartości - musi być przeprowadzony test Kruskala-Wallisa')
            use_ANOVA = False
        else:
            print('wszystkie grupy mają rozkład normalny, można skorzystać z testu ANOVA')

    print()

    testing_groups = []
    for row in df_to_collist(data_df): 
        testing_groups.append(pd.Series(row).dropna().to_numpy())

    if use_ANOVA:
        # Najpierw przeprowadzimy ANOA, który jest tzw. testem typu omnibus,
        # który sprawdza, że między którąkolwiek parą grup istnieje istotna statystycznie 
        # różnica średnich.
        print('Test ANOVA:')
        print()
        
        result_ANOVA = f_oneway(*testing_groups)
        print('\tstatystyka testu ANOVA\'a: %2.2f'%(result_ANOVA.statistic))
        print('\tp-wartość testu ANOVA\'a: %2.2f'%(result_ANOVA.pvalue))
        diary_info.update({'result_ANOVA':result_ANOVA})

        print()
        print('WYNIK TESTU NA RÓŻNICĘ ŚREDNICH GRUP: ',end='')
        if result_ANOVA.pvalue<significance_level:
            print('średnie przynajmniej jednej pary grup różnią się istotnie statystycznie, zaleca się wykonanie testu post-hoc')
            make_posthoc = True
        else:
            print('nie stwierdzono żadnych statystycznie istotnych różnic pomiędzy grupami na wejściu testu')
            make_posthoc = False
    else:
        # Najpierw przeprowadzimy Kruskala-Wallisa, który jest tzw. testem typu omnibus,
        # który sprawdza, że między którąkolwiek parą grup istnieje istotna statystycznie 
        # różnica median.
        print('Test Kruskala-Wallisa:')
        result_kruskal = kruskal(*df_to_collist(data_df))
        print('\tstatystyka testu Kruskala-Wallisa\'a: %2.2f'%(result_kruskal.statistic))
        print('\tp-wartość testu Kruskala-Wallisa\'a: %2.2f'%(result_kruskal.pvalue))
        diary_info.update({'result_Kruskal':result_kruskal})
        
        print()
        print('WYNIK TESTU NA RÓŻNICĘ MEDIAN GRUP: ',end='')
        if result_kruskal.pvalue<significance_level:
            print('mediany przynajmniej jednej pary grup różnią się istotnie statystycznie, zaleca się wykonanie testu post-hoc')
            make_posthoc = True
        else:
            print('nie stwierdzono żadnych statystycznie istotnych różnic pomiędzy grupami na wejściu testu')
            make_posthoc = False

    print()

    # test post-hoc wykonujemy tylko gdy wskazuje na to test 
    if make_posthoc:
        if use_ANOVA:
            print('Test post-hoc HSD Tukeya po teście ANOVA:')
            # Poniższa linijka wygeneruje nam macierz p-wartości dla testu post-hoc, 
            # dzięki czemu porównamy sobie równość każdej z badanych grup.
            
            # Test HSD-Tukeya zwraca wynik w postaci tabelki mówiącej o tym, jakie są wartości statystyki
            # dla każdej pary grup, jaka jest p-wartość i jaki jest przedział ufności różnic między innymi
            posthoc_mtx = pairwise_tukeyhsd(stat_data_lgfmt_values, stat_data_lgfmt_groups)
            print(posthoc_mtx)
            diary_info.update({'result_TukeyHSD':posthoc_mtx})
        else:
            print('Test post-hoc Dunn po teście Kruskala-Wallisa:')
            # W odróżnieniu od testu HSD-Tukeya, test Dunn zwraca tylko macierz p-wartości
            # mówiących o tym, czy dwie grupy różnią się od siebie istotnie statystycznie
            # Jest to nieco mniej informacji niż w teście HSD-Tukeya, bo nie ma informacji o przedziałach
            # ufności różnicy pomiędzy dwiema grupami
            # Implementacja posthoc_dunn sama potrafi sobie poradzić z 
            # wartościami np.nan w macierzy na jej wejściu - dlatego 
            # przekazujemy nieprzetworzoną ramkę danych DataFrame
            posthoc_mtx = (posthoc_dunn(data_df.values.T))
            print()
            # Wygodniej jednak będzie nam używać prawdziwych nazw grup, a nie ich numerków
            # wyznaczonych na bazie ich kolejności.
            # Najpierw jako kolumny przypiszmy nazwy grup
            posthoc_mtx = pd.DataFrame(posthoc_mtx)
            posthoc_mtx.columns = data_df.columns
            # Potem utwórzmy dodatkową kolumnę z nazwami grup dla wierszy
            posthoc_mtx = posthoc_mtx.assign(**{'nazwa grupy':data_df.columns}) # przypisanie nazwy grupy ze spacją niestety musi być w formacie **{'nazwa kolumny':wartości_kolumny, inaczej nie możemy użyć nazwy kolumny ze spacją w środku.

            # I ustawmy tę dodatkową kolumnę jako indeks, żeby wyświetlała się ona po 
            # lewej stronie macierzy zamiast numeru powiązanego z kolejnością grupy
            posthoc_mtx = posthoc_mtx.set_index('nazwa grupy')
            print(posthoc_mtx)
            diary_info.update({'result_Dunn':posthoc_mtx})
        print(' ')
        
        return diary_info