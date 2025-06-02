import random
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from collections import defaultdict

# Конфигурация
NUM_OPERATORS = 10
NUM_SHIFTS = 7
MAX_SHIFTS_PER_OPERATOR = 3

OPERATOR_SKILLS = {
    0: [0, 1, 2],     # покрывает shift 0, 3, 6
    1: [1, 4],     # shift 0, 2, 4, 5
    2: [3],     # shift 1, 3, 4, 5, 6
    3: [0, 3],     # shift 0, 1, 3, 4, 6
    4: [1, 2],     # shift 0, 1, 2, 5
    5: [4],     # shift 1, 2, 4, 5
    6: [0, 1, 3],  # shift 0, 1, 2, 3, 4, 6
    7: [3, 4],     # shift 1, 2, 4, 6
    8: [0, 2],     # shift 0, 1, 2, 3, 5
    9: [1]   # shift 0, 2, 4, 5, 6
}

PREFERENCES = {
    0: [0, 3],
    1: [0, 2],
    2: [1, 3],
    3: [1, 4],
    4: [2, 5],
    5: [2, 4],
    6: [3, 6],
    7: [4, 6],
    8: [1, 5],
    9: [2, 6]
}

SHIFT_TYPES = [
    [0, 1], [2, 1], [1, 4],
    [0, 2], [3, 4], [3, 2],
    [0, 3]
]

# Функции
def prefers_shift(operator, shift):
    return shift in PREFERENCES.get(operator, [])

def can_perform_shift(operator, shift):
    return all(skill in OPERATOR_SKILLS[operator] for skill in SHIFT_TYPES[shift])

# Генетическое кодирование: индивидуум — список списков операторов на каждую смену
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def generate_individual():
    individual = []
    for shift in range(NUM_SHIFTS):
        ops = random.sample(range(NUM_OPERATORS), random.randint(1, 5))
        individual.append(ops)
    return creator.Individual(individual)

toolbox.register("individual", generate_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    fitness = 0
    operator_shift_counts = defaultdict(int)

    for shift_id, operators in enumerate(individual):
        unique_ops = set(operators)

        for op in unique_ops:
            operator_shift_counts[op] += 1
            if prefers_shift(op, shift_id):
                fitness += 2
            if can_perform_shift(op, shift_id):
                fitness += 1
            else:
                fitness -= 3

        if len(unique_ops) > 6:
            fitness -= (len(unique_ops) - 6)

    for op, count in operator_shift_counts.items():
        if count > MAX_SHIFTS_PER_OPERATOR:
            fitness -= (count - MAX_SHIFTS_PER_OPERATOR) * 5

    return (fitness,)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
def mutate(individual):
    shift = random.randint(0, NUM_SHIFTS - 1)
    if random.random() < 0.5 and individual[shift]:
        individual[shift].pop(random.randint(0, len(individual[shift]) - 1))
    else:
        new_op = random.randint(0, NUM_OPERATORS - 1)
        if new_op not in individual[shift]:
            individual[shift].append(new_op)
    return (individual,)

toolbox.register("mutate", mutate)
toolbox.register("select", tools.selTournament, tournsize=3)

# Визуализация расписания
def visualize_schedule_tabular(schedule):
    fig, ax = plt.subplots()
    colors = plt.get_cmap("tab10")

    for shift_idx, assigned_operators in enumerate(schedule):
        for j, operator in enumerate(assigned_operators):
            ax.broken_barh([(j * 10, 9)], (shift_idx * 10, 9),
                           facecolors=colors(operator % 10 / 10))

    ax.set_yticks([10 * i + 5 for i in range(len(schedule))])
    ax.set_yticklabels([f"Смена {i + 1}" for i in range(len(schedule))])

    max_ops = max(len(op_list) for op_list in schedule)
    ax.set_xticks([10 * i + 5 for i in range(max_ops)])
    ax.set_xticklabels([f"Оператор {i + 1}" for i in range(max_ops)])

    ax.set_xlabel("Позиции в смене")
    ax.set_ylabel("Смены")
    ax.set_title("Распределение операторов по сменам (таблично)")
    plt.grid(False)
    plt.tight_layout()
    plt.show()


def plot_fitness_dynamics(avg_fitness, max_fitness):
    plt.plot(avg_fitness, label="Средний фитнес")
    plt.plot(max_fitness, label="Максимальный фитнес")
    plt.xlabel("Поколение")
    plt.ylabel("Фитнес")
    plt.title("Динамика обучения ГА")
    plt.legend()
    plt.grid(True)
    plt.show()

# Основной запуск
def main():
    random.seed(142)
    population = toolbox.population(n=200)
    NGEN = 100
    CXPB, MUTPB = 0.7, 0.3
    avg_fitness_history = []
    max_fitness_history = []

    for gen in range(NGEN):
        offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)
        for ind in offspring:
            ind.fitness.values = toolbox.evaluate(ind)
        population = toolbox.select(offspring, k=len(population))

        fitnesses = [ind.fitness.values[0] for ind in population]
        avg_fitness_history.append(sum(fitnesses) / len(fitnesses))
        max_fitness_history.append(max(fitnesses))

    best = tools.selBest(population, 1)[0]
    print("Лучшее расписание:", best)
    print("Фитнес:", best.fitness.values[0])
    visualize_schedule_tabular(best)
    plot_fitness_dynamics(avg_fitness_history, max_fitness_history)

if __name__ == "__main__":
    main()
