def save_res(number_of_task, number_of_q, answer):
    with open("../../../data/task" + str(number_of_task) + "/res/" + str(number_of_q) + ".txt", 'w') as f:
        f.write(str(answer))
