import random
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# Конфигурация задачи
NUM_OPERATORS = 10  # Количество операторов
SHIFTS_PER_DAY = 3  # Количество смен в день
NUM_SHIFTS = 7 * SHIFTS_PER_DAY  # Количество смен (например, 3 смены в день в течение недели)
MAX_SHIFTS_PER_OPERATOR = 3  # Максимальное количество смен на одного оператора в неделю

# Скиллы операторов (пример: 0 - умеет продавать валенки, 1 - умеет продавать диски, 2 - умеет продавать шины и т.д.)
OPERATOR_SKILLS = {
    0: [7, 5, 1, 2, 3],
    1: [1, 6, 8, 9, 4],
    2: [0, 9, 8, 7, 5],
    3: [1, 3, 9, 4, 2],
    4: [2, 4, 9, 5, 6],
    5: [4, 7, 0, 2, 6],
    6: [5, 0, 9, 4, 3],
    7: [6, 8, 3, 0, 2],
    8: [7, 9, 6, 0, 8],
    9: [1, 5, 7, 3, 8]
}

# Предпочтения операторов по сменам (пример)
PREFERENCES = {
    0: [0, 1, 2],
    1: [3, 4, 5],
    2: [6, 7, 8],
    3: [9, 10, 11],
    4: [12, 13, 14],
    5: [15, 16, 17],
    6: [18, 19, 20],
    7: [21, 0, 1],
    8: [2, 3, 4],
    9: [5, 6, 7]
}

# Типы смен. За одну смену могут быть востребованны разные навыки
SHIFT_TYPES = [
    [0, 1], [2, 3], [4, 5],  # День 1
    [6, 7], [8, 9], [0, 2],  # День 2
    [3, 5], [7, 9], [0, 3],  # День 3
    [2, 7], [5, 6], [1, 8],  # День 4
    [1, 5], [3, 9], [0, 4],  # День 5
    [0, 7], [4, 8], [1, 6],  # День 6
    [7, 8], [1, 3], [0, 8]   # День 7
]

def prefers_shift(operator, shift):
    return shift in PREFERENCES.get(operator, [])

def can_perform_shift(operator, shift):
    return all(skill in OPERATOR_SKILLS[operator] for skill in SHIFT_TYPES[shift])

# Функция оценки приспособленности
def evaluate(individual):
    fitness = 0

    # Проверка на повторения смен у одного оператора в один день
    for i in range(0, NUM_SHIFTS, SHIFTS_PER_DAY):
        day_shifts = individual[i:i + SHIFTS_PER_DAY]
        if len(set(day_shifts)) < len(day_shifts):  # Повторения смен у одного оператора
            fitness -= 1

    # Учет предпочтений сотрудников и возможности выполнения смены
    for shift in range(NUM_SHIFTS):
        operator = individual[shift]
        if prefers_shift(operator, shift):
            fitness += 1
        if not can_perform_shift(operator, shift):
            fitness -= 5

    # Учет равномерности распределения смен
    shifts_per_operator = [0] * NUM_OPERATORS
    for operator in individual:
        shifts_per_operator[operator] += 1

    for shifts in shifts_per_operator:
        if shifts > MAX_SHIFTS_PER_OPERATOR:  # Превышение максимального количества смен для одного оператора
            fitness -= (shifts - MAX_SHIFTS_PER_OPERATOR) * 5

    return (fitness,)

# Настройка генетического алгоритма
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 0, NUM_OPERATORS - 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, NUM_SHIFTS)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# Основной цикл алгоритма
def main():
    random.seed(23)                         # Установка начального значения для генератора случайных чисел для воспроизводимости результатов.
    population = toolbox.population(n=300)  # Создание начальной популяции из 300 индивидов.
    NGEN = 450                              # Количество поколений для выполнения генетического алгоритма.
    CXPB, MUTPB = 1, 1                  # Вероятности кроссовера и мутации соответственно.

    for gen in range(NGEN):
        offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)
        fits = map(toolbox.evaluate, offspring)

        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit

        population = toolbox.select(offspring, k=len(population))
        
        # Отображение лучшего результата каждого поколения
        best_ind = tools.selBest(population, 1)[0]
        print(f"Поколение {gen}: Лучший результат: {best_ind}, Фитнес: {best_ind.fitness.values}")

    best_ind = tools.selBest(population, 1)[0]
    print("Лучший результат: %s, %s" % (best_ind, best_ind.fitness.values))
    visualize_schedule(best_ind)

def visualize_schedule(schedule):
    fig, ax = plt.subplots()
    colors = plt.get_cmap("tab10")

    for shift in range(NUM_SHIFTS):
        operator = schedule[shift]
        day = shift // SHIFTS_PER_DAY
        shift_in_day = shift % SHIFTS_PER_DAY

        ax.broken_barh([(shift_in_day * 10, 10)], (day * 10, 9), facecolors=colors(operator / NUM_OPERATORS))

    ax.set_yticks([10 * i + 5 for i in range(7)])
    ax.set_yticklabels([f"День {i+1}" for i in range(7)])
    ax.set_xticks([10 * i + 5 for i in range(SHIFTS_PER_DAY)])
    ax.set_xticklabels(["Смена 1", "Смена 2", "Смена 3"])
    ax.set_xlabel("Смен")
    ax.set_ylabel("Дней")
    ax.set_title("График работы колл-центра")
    plt.show()

if __name__ == "__main__":
    main()