import random
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# Конфигурация задачи
NUM_OPERATORS = 10
SHIFTS_PER_DAY = 3
NUM_SHIFTS = 7 * SHIFTS_PER_DAY
MAX_SHIFTS_PER_OPERATOR = 3
NUM_SKILLS = 10
SKILLS_PER_OPERATOR = 5
PREFERENCES_PER_OPERATOR = 3
SKILLS_PER_SHIFT = 2

# Генерация OPERATOR_SKILLS
OPERATOR_SKILLS = {
    i: random.sample(range(NUM_SKILLS), SKILLS_PER_OPERATOR)
    for i in range(NUM_OPERATORS)
}

# Генерация PREFERENCES
PREFERENCES = {
    i: random.sample(range(NUM_SHIFTS), PREFERENCES_PER_OPERATOR)
    for i in range(NUM_OPERATORS)
}

# Генерация SHIFT_TYPES
SHIFT_TYPES = [
    random.sample(range(NUM_SKILLS), SKILLS_PER_SHIFT)
    for _ in range(NUM_SHIFTS)
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

def random_scheduler():
    return [random.randint(0, NUM_OPERATORS - 1) for _ in range(NUM_SHIFTS)]

def greedy_scheduler():
    schedule = [-1] * NUM_SHIFTS
    shifts_per_operator = [0] * NUM_OPERATORS
    for shift in range(NUM_SHIFTS):
        for operator in range(NUM_OPERATORS):
            if can_perform_shift(operator, shift) and shifts_per_operator[operator] < MAX_SHIFTS_PER_OPERATOR:
                schedule[shift] = operator
                shifts_per_operator[operator] += 1
                break
        if schedule[shift] == -1:
            schedule[shift] = random.randint(0, NUM_OPERATORS - 1)
    return schedule

def plot_schedule(schedule, title):
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
    ax.set_title(title)

def plot_fitness_dynamics(avg_fitness_history, max_fitness_history):
    plt.plot(avg_fitness_history, label="Средняя приспособленность")
    plt.plot(max_fitness_history, label="Максимальная приспособленность")
    plt.xlabel("Поколение")
    plt.ylabel("Приспособленность")
    plt.title("Динамика обучения ГА")
    plt.legend()
    plt.grid()
    plt.show()

def compare_with_heuristics(best_individual, greedy, rand):
    rand_fitness = evaluate(rand)
    greedy_fitness = evaluate(greedy)
    ga_fitness = evaluate(best_individual)
    print("\n📊 Сравнение подходов:")
    print(f"Генетический алгоритм: {ga_fitness[0]}")
    print(f"Жадное решение:         {greedy_fitness[0]}")
    print(f"Случайное решение:     {rand_fitness[0]}")

def compare_algorithms_plot(ga_score, greedy_score, random_score):
    labels = ['Генетический', 'Жадный', 'Случайный']
    scores = [ga_score, greedy_score, random_score]
    plt.bar(labels, scores, color=['green', 'orange', 'red'])
    plt.ylabel("Фитнес")
    plt.title("Сравнение подходов")
    plt.show()

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
        best_ind = tools.selBest(population, 1)[0]
        avg_fit = sum(ind.fitness.values[0] for ind in population) / len(population)
        avg_fitness_history.append(avg_fit)
        max_fitness_history.append(best_ind.fitness.values[0])
        print(f"Поколение {gen}: Лучший результат: {best_ind}, Фитнес: {best_ind.fitness.values}")

    best_ind = tools.selBest(population, 1)[0]
    #print("Лучший результат ГА:", best_ind, "Приспособленность:", best_ind.fitness.values[0])
    #plot_schedule(best_ind, "ГА: Лучший график")
    #plot_fitness_dynamics(avg_fitness_history, max_fitness_history)

    greedy = greedy_scheduler()
    random_sched = random_scheduler()

    #print("Жадный алгоритм:", greedy, "Приспособленность:", evaluate(greedy)[0])
    #print("Случайный алгоритм:", random_sched, "Приспособленность:", evaluate(random_sched)[0])
    compare_with_heuristics(best_ind, greedy, random_sched)

    plot_schedule(best_ind, "Генетический алгоритм")
    plot_schedule(greedy, "Жадный алгоритм")
    plot_schedule(random_sched, "Случайный алгоритм")
    plt.show()
    compare_algorithms_plot(best_ind.fitness.values[0], evaluate(greedy)[0], evaluate(random_sched)[0])

if __name__ == "__main__":
    main()