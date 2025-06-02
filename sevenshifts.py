import random
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# Конфигурация задачи
NUM_OPERATORS = 10
NUM_SHIFTS = 7
SHIFT_OPERATOR_REQUIREMENTS = [2, 3, 1, 2, 3, 2, 1]  # Кол-во операторов на каждую смену
MAX_SHIFTS_PER_OPERATOR = 3

OPERATOR_SKILLS = {
    0: [7, 3, 1, 0, 9],
    1: [5, 6, 0, 2, 4],
    2: [2, 1, 6, 9, 3],
    3: [1, 8, 7, 4, 2],
    4: [3, 6, 5, 7, 0],
    5: [2, 4, 9, 3, 5],
    6: [6, 1, 7, 8, 0],
    7: [8, 9, 5, 2, 3],
    8: [4, 0, 1, 6, 8],
    9: [7, 2, 9, 0, 5]
}

PREFERENCES = {
    0: [0, 2],
    1: [1, 5],
    2: [3],
    3: [0, 1],
    4: [2, 4],
    5: [1, 6],
    6: [5],
    7: [3, 4],
    8: [6],
    9: [0, 1, 2]
}

SHIFT_TYPES = [
    [6, 7], [3, 2], [8, 4],
    [0, 9], [2, 5], [1, 7],
    [4, 0]
]

# Вспомогательные функции
def prefers_shift(operator, shift):
    return shift in PREFERENCES.get(operator, [])

def can_perform_shift(operator, shift):
    return all(skill in OPERATOR_SKILLS[operator] for skill in SHIFT_TYPES[shift])

def reshape_schedule(flat_schedule):
    schedule = []
    idx = 0
    for count in SHIFT_OPERATOR_REQUIREMENTS:
        schedule.append(flat_schedule[idx:idx + count])
        idx += count
    return schedule

def flatten_schedule(nested_schedule):
    return [op for shift in nested_schedule for op in shift]

# Оценка индивидуума
def evaluate(individual):
    schedule = reshape_schedule(individual)
    fitness = 0

    shifts_per_operator = [0] * NUM_OPERATORS
    for shift_idx, operators in enumerate(schedule):
        if len(set(operators)) < len(operators):
            fitness -= 3  # Повтор операторов в одной смене

        for op in operators:
            if prefers_shift(op, shift_idx):
                fitness += 1
            if not can_perform_shift(op, shift_idx):
                fitness -= 5
            shifts_per_operator[op] += 1

    for count in shifts_per_operator:
        if count > MAX_SHIFTS_PER_OPERATOR:
            fitness -= (count - MAX_SHIFTS_PER_OPERATOR) * 5

    return (fitness,)

# Настройка DEAP
TOTAL_ASSIGNMENTS = sum(SHIFT_OPERATOR_REQUIREMENTS)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 0, NUM_OPERATORS - 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, TOTAL_ASSIGNMENTS)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# Визуализация
def visualize_schedule(flat_schedule):
    schedule = reshape_schedule(flat_schedule)
    fig, ax = plt.subplots()
    colors = plt.get_cmap("tab10")

    for shift_idx, assigned_operators in enumerate(schedule):
        for j, operator in enumerate(assigned_operators):
            ax.broken_barh([(j * 10, 9)], (shift_idx * 10, 9), facecolors=colors(operator / NUM_OPERATORS))

    ax.set_yticks([10 * i + 5 for i in range(len(schedule))])
    ax.set_yticklabels([f"Смена {i + 1}" for i in range(len(schedule))])
    ax.set_xticks([10 * i + 5 for i in range(max(len(op_list) for op_list in schedule))])
    ax.set_xticklabels([f"Оператор {i + 1}" for i in range(max(len(op_list) for op_list in schedule))])
    ax.set_xlabel("Операторы в смене")
    ax.set_ylabel("Смены")
    ax.set_title("Распределение операторов по сменам")
    plt.tight_layout()
    plt.show()

def plot_fitness_dynamics(avg_fitness, max_fitness, label_prefix=""):
    plt.plot(avg_fitness, label=f"{label_prefix}Средняя")
    plt.plot(max_fitness, label=f"{label_prefix}Макс")
    plt.xlabel("Поколение")
    plt.ylabel("Фитнес")
    plt.title("Динамика обучения ГА")
    plt.legend()
    plt.grid(True)
    plt.show()

def run_experiment(cxpb, mutpb, ngen=100):
    population = toolbox.population(n=300)
    avg_fitness_history = []
    max_fitness_history = []
    for gen in range(ngen):
        offspring = algorithms.varAnd(population, toolbox, cxpb=cxpb, mutpb=mutpb)
        fits = list(map(toolbox.evaluate, offspring))
        for ind, fit in zip(offspring, fits):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))
        fitnesses = [ind.fitness.values[0] for ind in population]
        avg_fitness_history.append(sum(fitnesses) / len(fitnesses))
        max_fitness_history.append(max(fitnesses))
    best = tools.selBest(population, 1)[0]
    return best, avg_fitness_history, max_fitness_history

def main():
    random.seed(42)
    best, avg_hist, max_hist = run_experiment(cxpb=1, mutpb=1)
    print("Лучший индивидуум:", reshape_schedule(best))
    visualize_schedule(best)
    plot_fitness_dynamics(avg_hist, max_hist)

if __name__ == "__main__":
    main()
