import random
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# Конфигурация задачи
NUM_OPERATORS = 10
SHIFTS_PER_DAY = 3
NUM_SHIFTS = 21
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
    0: [4, 8, 19],
    1: [0, 11, 14],
    2: [6, 2, 18],
    3: [12, 15, 5],
    4: [9, 7, 10],
    5: [3, 13, 20],
    6: [17, 1, 16],
    7: [19, 8, 6],
    8: [7, 2, 0],
    9: [10, 5, 14]
}

SHIFT_TYPES = [
    [6, 7], [3, 2], [8, 4],
    [0, 9], [2, 5], [1, 7],
    [4, 0], [6, 3], [5, 1],
    [7, 2], [8, 0], [9, 6],
    [3, 5], [1, 4], [2, 7],
    [0, 6], [3, 8], [5, 9],
    [1, 2], [4, 7], [6, 0]
]

def prefers_shift(operator, shift):
    return shift in PREFERENCES.get(operator, [])

def can_perform_shift(operator, shift):
    return all(skill in OPERATOR_SKILLS[operator] for skill in SHIFT_TYPES[shift])

def evaluate(individual):
    fitness = 0
    for i in range(0, NUM_SHIFTS, SHIFTS_PER_DAY):
        day_shifts = individual[i:i + SHIFTS_PER_DAY]
        if len(set(day_shifts)) < len(day_shifts):
            fitness -= 1

    for shift in range(NUM_SHIFTS):
        operator = individual[shift]
        if prefers_shift(operator, shift):
            fitness += 1
        if not can_perform_shift(operator, shift):
            fitness -= 5

    shifts_per_operator = [0] * NUM_OPERATORS
    for operator in individual:
        shifts_per_operator[operator] += 1

    for shifts in shifts_per_operator:
        if shifts > MAX_SHIFTS_PER_OPERATOR:
            fitness -= (shifts - MAX_SHIFTS_PER_OPERATOR) * 5

    return (fitness,)

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

def random_schedule():
    return [random.randint(0, NUM_OPERATORS - 1) for _ in range(NUM_SHIFTS)]

def plot_fitness_dynamics(avg_fitness, max_fitness, label_prefix=""):
    plt.plot(avg_fitness, label=f"{label_prefix}Средняя")
    plt.plot(max_fitness, label=f"{label_prefix}Макс")
    plt.xlabel("Поколение")
    plt.ylabel("Фитнес")
    plt.title("Динамика обучения ГА")
    plt.legend()
    plt.grid(True)

def experiment_suite():
    configs = [
        {"cxpb": 0.5, "mutpb": 0.1, "label": "CX=0.5, MUT=0.1"},
        {"cxpb": 0.9, "mutpb": 0.05, "label": "CX=0.9, MUT=0.05"},
        {"cxpb": 0.7, "mutpb": 0.2, "label": "CX=0.7, MUT=0.2"},
        {"cxpb": 1, "mutpb": 1, "label": "CX=1.0, MUT=1.0"},
    ]
    plt.figure(figsize=(10, 6))
    for config in configs:
        best, avg_hist, max_hist = run_experiment(config["cxpb"], config["mutpb"])
        plot_fitness_dynamics(avg_hist, max_hist, label_prefix=f"{config['label']} ")
    plt.show()

def run_experiment(cxpb, mutpb, ngen=100):
    random.seed(126)
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
    random.seed(126)
    population = toolbox.population(n=300)
    NGEN = 100
    CXPB, MUTPB = 1, 1
    avg_fitness_history = []
    max_fitness_history = []
    for gen in range(NGEN):
        offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)
        fits = list(map(toolbox.evaluate, offspring))
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))
        fitnesses = [ind.fitness.values[0] for ind in population]
        avg_fitness_history.append(sum(fitnesses) / len(fitnesses))
        max_fitness_history.append(max(fitnesses))
        best_ind = tools.selBest(population, 1)[0]
        print(f"Поколение {gen}: Лучший результат: {best_ind}, Фитнес: {best_ind.fitness.values}")
    best_ind = tools.selBest(population, 1)[0]
    print("Лучший результат: %s, %s" % (best_ind, best_ind.fitness.values))
    visualize_schedule(best_ind)
    plot_fitness_dynamics(avg_fitness_history, max_fitness_history)
    experiment_suite()

if __name__ == "__main__":
    main()
